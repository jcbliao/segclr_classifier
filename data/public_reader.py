"""Vendored + wrapped access to the public SegCLR embedding release.

Vendors just the two functions/classes this project needs from
google-research/connectomics (Apache-2.0) -- EmbeddingReader and md5_shard --
instead of `pip install`ing the full package, which pulls in tensorflow/edward2
for the (unrelated) SNGP classification submodule this project does not reuse
(see CLAUDE.md: the GNN classifier here is a from-scratch design, not a
reimplementation of the original BERT+SNGP classifier). Source, retrieved
2026-08-04, Apache License 2.0:
  https://github.com/google-research/connectomics/blob/main/connectomics/segclr/reader.py
  https://github.com/google-research/connectomics/blob/main/connectomics/common/sharding.py

Colab gists this module's design is derived from:
  https://colab.research.google.com/gist/chinasaur/63f15b3f37b35b5bb27de31ba0a0087f
  https://colab.research.google.com/gist/chinasaur/47b631677d099fa8059a7ce7c323222b
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def md5_shard(
    segment_id: int, num_shards: int, byteorder: str = "little", bytewidth: int = 8
) -> int:
    """Vendored from connectomics.common.sharding.md5_shard."""
    md5 = hashlib.md5()
    md5.update(segment_id.to_bytes(bytewidth, byteorder))
    return int.from_bytes(md5.digest(), byteorder) % num_shards


class EmbeddingReader:
    """Vendored from connectomics.segclr.reader.EmbeddingReader.

    Reads one segment's embeddings from a CSV-inside-ZIP shard. Each CSV row is
    `node_id,x,y,z,e_0,...,e_{D-1}`; the reader keys the returned dict by the
    (x, y, z) tuple (float, in whatever coordinate system the dataset key
    encodes) rather than node_id -- node_id here is Google's internal
    skeletonization index, not a CAVE skeleton node_id. Those get reconciled by
    nearest-neighbor xyz matching in `ingest_public_microns.py`, never by
    reusing this id directly.
    """

    def __init__(self, filesystem, zipdir: str, sharder):
        self._filesystem = filesystem
        self._zipdir = zipdir
        self._sharder = sharder

    def _get_csv_data(self, seg_id: int) -> str:
        shard = self._sharder(seg_id)
        zip_path = os.path.join(self._zipdir, f"{shard}.zip")
        with self._filesystem.open(zip_path) as f:
            with zipfile.ZipFile(f) as z:
                with z.open(f"{seg_id}.csv") as c:
                    return c.read().decode("utf-8")

    def _parse_csv_data(self, csv_data: str) -> Mapping[tuple[float, float, float], list[float]]:
        embeddings_from_xyz = {}
        for line in csv_data.split("\n"):
            if not line:
                continue
            fields = line.split(",")
            xyz = tuple(float(f) for f in fields[1:4])
            embedding = [float(f) for f in fields[4:]]
            assert xyz not in embeddings_from_xyz, f"duplicate xyz {xyz} for seg {seg_id}"
            embeddings_from_xyz[xyz] = embedding
        return embeddings_from_xyz

    def __getitem__(self, seg_id: int):
        return self._parse_csv_data(self._get_csv_data(seg_id))


# Bucket layout for the "bytewidth64" export round -- the only one released so
# far (see upstream comment: newer exports are meant to switch to bytewidth 8,
# but haven't yet). Keys this project doesn't use are kept for completeness --
# e.g. h01 is a candidate second dataset for the norm diagnostic later.
DATA_URL_FROM_KEY_BYTEWIDTH64 = dict(
    h01="gs://h01-release/data/20220326/c3/embeddings/segclr_csvzips",
    h01_nm_coord="gs://h01-release/data/20220326/c3/embeddings/segclr_nm_coord_csvzips",
    h01_agg10um="gs://h01-release/data/20220326/c3/embeddings/segclr_aggregated_10um_csvzips",
    microns_v343="gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_csvzips",
    microns_nm_coord_public_offset_v343=(
        "gs://iarpa_microns/minnie/minnie65/embeddings_m343/"
        "segclr_nm_coord_public_offset_csvzips"
    ),
    microns_v343_agg25um=(
        "gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_aggregated_25um_csvzips"
    ),
    microns_v117="gs://iarpa_microns/minnie/minnie65/embeddings/segclr_csvzips",
)

# Google's own aggregated bucket paired with the nm/public-offset coordinate
# frame is not in DATA_URL_FROM_KEY_BYTEWIDTH64 (only plain `_agg25um`, in the
# internal coordinate frame, is) -- the classifier gist builds this path by
# hand with a custom sharder. Recorded here for completeness only: this
# project does NOT read from it. segclr-db recomputes the 25um geodesic-mean
# baseline itself from the raw embeddings we ingest (src/aggregate.py), so the
# baseline and the GNN provably consume the exact same rows for the exact same
# cells -- pulling Google's separately-computed aggregate would not guarantee
# that.
MICRONS_NM_COORD_PUBLIC_OFFSET_AGG25UM = (
    "gs://iarpa_microns/minnie/minnie65/embeddings_m343/"
    "segclr_nm_coord_public_offset_aggregated_25um_csvzips"
)

MICRONS_CELLTYPE_LABELS_URL = (
    "gs://iarpa_microns/minnie/minnie65/embedding_classification/"
    "training_data/labeled_cell_m343_df_221011b.feather"
)


def get_reader(key: str, filesystem, num_shards: int = 10_000) -> EmbeddingReader:
    """Vendored from connectomics.segclr.reader.get_reader."""
    if key not in DATA_URL_FROM_KEY_BYTEWIDTH64:
        raise ValueError(f"Key not found: {key}")
    url = DATA_URL_FROM_KEY_BYTEWIDTH64[key]
    bytewidth = 64

    def sharder(segment_id: int) -> int:
        return md5_shard(segment_id, num_shards=num_shards, bytewidth=bytewidth)

    return EmbeddingReader(filesystem, url, sharder)


@dataclass
class RawCellEmbeddings:
    """One cell's raw per-node SegCLR embeddings, as arrays instead of a dict."""

    root_id: int
    xyz_nm: np.ndarray  # (n, 3) float32, nm coords in the public-offset CAVE frame
    embeddings: np.ndarray  # (n, D) float32


def get_public_filesystem():
    """Anonymous GCS access -- the public release needs no credentials."""
    import gcsfs

    return gcsfs.GCSFileSystem(token="anon")


def get_raw_cell_embeddings(
    root_id: int,
    filesystem=None,
    data_key: str = "microns_nm_coord_public_offset_v343",
) -> RawCellEmbeddings:
    """Fetch one cell's raw (unaggregated) embeddings, keyed by nm xyz.

    `data_key` defaults to the nm-coord, public-offset variant so the returned
    xyz lines up with CAVE skeleton coordinates (both are nm in the public
    materialization's frame). Plain `microns_v343` is in the internal
    segmentation pipeline's own coordinate frame and will NOT line up with CAVE
    skeletons -- do not swap this default without re-deriving the offset.
    """
    filesystem = filesystem or get_public_filesystem()
    reader = get_reader(data_key, filesystem)
    embeddings_from_xyz = reader[root_id]
    if not embeddings_from_xyz:
        return RawCellEmbeddings(
            root_id, np.zeros((0, 3), np.float32), np.zeros((0, 0), np.float32)
        )
    xyz = np.array(list(embeddings_from_xyz.keys()), dtype=np.float32)
    embeddings = np.array(list(embeddings_from_xyz.values()), dtype=np.float32)
    return RawCellEmbeddings(root_id, xyz, embeddings)


def get_celltype_labels(filesystem=None) -> pd.DataFrame:
    """Fetch the official ground-truth cell-type label table.

    Columns are whatever the release actually has (at minimum `seg_id`,
    `cell_type` -- those are the only two the original classifier gist reads).
    Callers should log `.columns`/`.dtypes` the first time rather than assuming
    a richer schema we haven't confirmed.
    """
    import pandas as pd

    filesystem = filesystem or get_public_filesystem()
    with filesystem.open(MICRONS_CELLTYPE_LABELS_URL, "rb") as f:
        return pd.read_feather(f)
