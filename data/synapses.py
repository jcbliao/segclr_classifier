"""CAVE synapse queries for the cells in ``data/manifest.json``.

CAVE's synapse table states polarity explicitly: every row carries both
``pre_pt_root_id`` (the presynaptic cell, i.e. the bouton side) and
``post_pt_root_id`` (the postsynaptic cell), plus a point on each partner and
the cleft centroid. Nothing here infers direction from geometry.

Two views of that table, one per direction, are what
:mod:`data.build_synapses` materialises:

``outgoing``
    rows where one of our cells is ``pre_pt_root_id`` -- its **presynaptic
    locations**, the places where it synapses onto something else.
``incoming``
    rows where one of our cells is ``post_pt_root_id`` -- its **postsynaptic
    locations**, each carrying the presynaptic partner's root_id.

Both are stored with identical columns (see :data:`SCHEMA`), so they can be
concatenated or compared without a translation step. ``cell_*`` is always the
side belonging to our cell and ``partner_*`` the other side, which means
``partner_root_id`` is the postsynaptic partner in ``outgoing`` and the
presynaptic partner in ``incoming``.

Why not ``segclr_db.cave.query_synapse_partners``
-------------------------------------------------
That function deduplicates to one representative synapse per partner
supervoxel, because what it feeds is a per-partner embedding. Deduplicating
would delete synaptic *locations* on our own cell, which is the thing being
asked for here, so this module queries the table directly instead.

Three things that are silent when wrong
---------------------------------------
1. **Positions come back in voxels unless you ask for nanometres.**
   ``synapses_pni_2`` is stored at (4, 4, 40) nm/voxel; ``query_table``
   returns positions in the table's own resolution unless
   ``desired_resolution=[1, 1, 1]`` is passed, and nothing about the returned
   column names records which one you got. Every query here passes it, and
   :func:`normalize_frame` rejects a frame whose resolution wasn't converted.
2. **A truncated query looks exactly like a small one.** The server caps rows
   per query and announces truncation only through a logged ``Warning``
   header. So every fetch asks for ``get_counts`` first and refuses to return a
   frame whose length does not equal that count.
3. **root_ids are materialization-scoped.** Synapses must be queried at the
   same ``mat_version`` the rest of this project uses (1718), or the returned
   root_ids name cells from a different snapshot of the proofreading -- the
   same failure mode hard constraint #8 covers.

``partner_root_id == 0`` means the partner supervoxel resolved to no cell at
this materialization. Those rows are kept: the synapse is still a real location
on our cell, and dropping them would silently shrink the location database.
Filter on it downstream when the partner is what matters.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)

#: The public datastack this account can read, and the materialization the rest
#: of the project is built at (``data/manifest.json``'s ``label_mat_version``).
#: ``minnie65_phase3_v1`` returns 403 for this account -- see
#: ``scripts/check_cell_type_labels.py``.
DEFAULT_DATASTACK = "minnie65_public"
DEFAULT_MAT_VERSION = 1718
DEFAULT_SYNAPSE_TABLE = "synapses_pni_2"

#: Which CAVE column selects a mode, and which prefix is then *our* cell.
#: ``outgoing`` filters on ``pre_pt_root_id`` because our cell is the
#: presynaptic one there; the partner is read off the opposite prefix.
MODES: dict[str, dict[str, str]] = {
    "outgoing": {"cell": "pre_pt", "partner": "post_pt"},
    "incoming": {"cell": "post_pt", "partner": "pre_pt"},
}

#: Rows above this in a single query are split into smaller root_id groups
#: rather than risking the server's own cap. Well above any single cell: the
#: busiest MICrONS neurons have a few tens of thousands of synapses per side.
MAX_ROWS_PER_QUERY = 200_000

SCHEMA = pa.schema(
    [
        # our cell -- the one from data/manifest.json this row belongs to
        pa.field("cell_root_id", pa.int64(), nullable=False),
        # "outgoing" (cell is presynaptic) | "incoming" (cell is postsynaptic)
        pa.field("mode", pa.string(), nullable=False),
        # CAVE's own synapse id, stable within a materialization
        pa.field("synapse_id", pa.int64(), nullable=False),
        # 0 where the partner supervoxel resolved to no cell -- kept, not dropped
        pa.field("partner_root_id", pa.int64(), nullable=False),
        pa.field("cell_supervoxel_id", pa.int64(), nullable=False),
        pa.field("partner_supervoxel_id", pa.int64(), nullable=False),
        # CAVE's `size`: cleft segmentation voxel count
        pa.field("cleft_size", pa.int64()),
        # the synaptic point on OUR cell (pre_pt for outgoing, post_pt for incoming)
        pa.field("cell_x_nm", pa.float32(), nullable=False),
        pa.field("cell_y_nm", pa.float32(), nullable=False),
        pa.field("cell_z_nm", pa.float32(), nullable=False),
        # the same point on the partner
        pa.field("partner_x_nm", pa.float32(), nullable=False),
        pa.field("partner_y_nm", pa.float32(), nullable=False),
        pa.field("partner_z_nm", pa.float32(), nullable=False),
        # the cleft centroid: "where the synapse is", independent of side
        pa.field("ctr_x_nm", pa.float32(), nullable=False),
        pa.field("ctr_y_nm", pa.float32(), nullable=False),
        pa.field("ctr_z_nm", pa.float32(), nullable=False),
    ]
)

EMPTY = SCHEMA.empty_table().to_pandas()


def build_client(token: str, datastack: str = DEFAULT_DATASTACK, mat_version: int = DEFAULT_MAT_VERSION):
    """A CAVEclient pinned to the materialization the project is built at."""
    import caveclient

    return caveclient.CAVEclient(datastack, auth_token=token, version=mat_version)


def positions_nm(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """``(N, 3)`` nanometre coordinates for ``pre_pt`` / ``post_pt`` / ``ctr_pt``.

    ``query_table`` splits position columns into ``_x``/``_y``/``_z`` on the
    static path but not on every path, so both shapes are accepted rather than
    assumed.
    """
    split = [f"{prefix}_position_{axis}" for axis in "xyz"]
    if all(col in frame.columns for col in split):
        return frame[split].to_numpy(dtype="float64")
    packed = f"{prefix}_position"
    if packed in frame.columns:
        return np.vstack([np.asarray(p, dtype="float64") for p in frame[packed]])
    raise KeyError(f"no {prefix}_position columns in {sorted(frame.columns)}")


def normalize_frame(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """CAVE's synapse rows -> :data:`SCHEMA`, with our cell on the ``cell_*`` side."""
    spec = MODES[mode]
    cell, partner = spec["cell"], spec["partner"]

    cell_xyz = positions_nm(frame, cell)
    partner_xyz = positions_nm(frame, partner)
    ctr_xyz = positions_nm(frame, "ctr_pt")

    out = pd.DataFrame(
        {
            "cell_root_id": frame[f"{cell}_root_id"].to_numpy(dtype="int64"),
            "mode": mode,
            "synapse_id": frame["id"].to_numpy(dtype="int64"),
            "partner_root_id": frame[f"{partner}_root_id"].to_numpy(dtype="int64"),
            "cell_supervoxel_id": frame[f"{cell}_supervoxel_id"].to_numpy(dtype="int64"),
            "partner_supervoxel_id": frame[f"{partner}_supervoxel_id"].to_numpy(dtype="int64"),
            "cleft_size": frame["size"].to_numpy(dtype="int64") if "size" in frame else -1,
            "cell_x_nm": cell_xyz[:, 0].astype("float32"),
            "cell_y_nm": cell_xyz[:, 1].astype("float32"),
            "cell_z_nm": cell_xyz[:, 2].astype("float32"),
            "partner_x_nm": partner_xyz[:, 0].astype("float32"),
            "partner_y_nm": partner_xyz[:, 1].astype("float32"),
            "partner_z_nm": partner_xyz[:, 2].astype("float32"),
            "ctr_x_nm": ctr_xyz[:, 0].astype("float32"),
            "ctr_y_nm": ctr_xyz[:, 1].astype("float32"),
            "ctr_z_nm": ctr_xyz[:, 2].astype("float32"),
        }
    )
    _assert_nanometres(ctr_xyz)
    return out


def _assert_nanometres(xyz: np.ndarray) -> None:
    """Catch a frame that came back in voxels despite ``desired_resolution``.

    The MICrONS volume is ~1.4 x 1.1 x 0.9 mm, so real nanometre coordinates run
    to ~1e6 while (4, 4, 40) nm voxel indices stay under ~3e5 in x/y. A whole
    chunk sitting below the voxel ceiling is the signature of an unconverted
    frame, and there is nothing in the column names to distinguish the two.
    """
    if len(xyz) == 0:
        return
    if float(np.max(xyz[:, 0])) < 400_000.0 and float(np.max(xyz[:, 1])) < 400_000.0:
        raise ValueError(
            "synapse positions look like voxels, not nanometres "
            f"(max x={np.max(xyz[:, 0]):.0f}, max y={np.max(xyz[:, 1]):.0f}); "
            "desired_resolution=[1,1,1] did not take effect"
        )


def count_synapses(
    client,
    root_ids: Sequence[int],
    mode: str,
    synapse_table: str = DEFAULT_SYNAPSE_TABLE,
) -> int:
    """How many rows the matching query would return, without returning them."""
    result = client.materialize.query_table(
        synapse_table,
        filter_in_dict={f"{MODES[mode]['cell']}_root_id": [int(r) for r in root_ids]},
        get_counts=True,
    )
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return 0
        column = "count" if "count" in result.columns else result.columns[0]
        return int(result[column].iloc[0])
    if isinstance(result, dict):
        return int(next(iter(result.values())))
    return int(result)


def _query_once(client, root_ids: Sequence[int], mode: str, synapse_table: str) -> pd.DataFrame:
    return client.materialize.query_table(
        synapse_table,
        filter_in_dict={f"{MODES[mode]['cell']}_root_id": [int(r) for r in root_ids]},
        split_positions=True,
        desired_resolution=[1, 1, 1],
        metadata=False,
    )


def _retry(fn, *, attempts: int = 4, base_delay: float = 5.0, what: str = "query"):
    """Retry a CAVE call on transient failure, with linear backoff.

    Linear rather than exponential, and only a handful of attempts: this runs
    against a service other labs share, so a persistent failure should surface
    as an error to look at rather than as a long quiet retry storm.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - caveclient raises many types
            last = exc
            if attempt == attempts:
                break
            delay = base_delay * attempt
            logger.warning("%s failed (attempt %d/%d): %s -- retrying in %.0fs", what, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {attempts} attempts") from last


def fetch_synapses(
    client,
    root_ids: Sequence[int],
    mode: str,
    synapse_table: str = DEFAULT_SYNAPSE_TABLE,
    sleep_s: float = 0.0,
    max_rows: int = MAX_ROWS_PER_QUERY,
) -> pd.DataFrame:
    """Every synapse of ``root_ids`` on the ``mode`` side, normalized to :data:`SCHEMA`.

    The count is fetched first and the returned frame checked against it, so a
    server-side truncation raises instead of quietly writing a short shard. A
    group whose count is too large is split in half and recursed on; a *single*
    cell over the cap is a real outlier and raises rather than being paginated,
    since offset pagination would need an ordering guarantee the endpoint does
    not give.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {', '.join(MODES)}")
    root_ids = [int(r) for r in root_ids]
    if not root_ids:
        return EMPTY.copy()

    expected = _retry(
        lambda: count_synapses(client, root_ids, mode, synapse_table),
        what=f"count {mode} for {len(root_ids)} cells",
    )
    if sleep_s:
        time.sleep(sleep_s)
    if expected == 0:
        return EMPTY.copy()

    if expected > max_rows:
        if len(root_ids) == 1:
            raise RuntimeError(
                f"cell {root_ids[0]} has {expected} {mode} synapses, over the "
                f"{max_rows}-row single-query cap"
            )
        half = len(root_ids) // 2
        left = fetch_synapses(client, root_ids[:half], mode, synapse_table, sleep_s, max_rows)
        right = fetch_synapses(client, root_ids[half:], mode, synapse_table, sleep_s, max_rows)
        return pd.concat([left, right], ignore_index=True)

    frame = _retry(
        lambda: _query_once(client, root_ids, mode, synapse_table),
        what=f"query {mode} for {len(root_ids)} cells",
    )
    if sleep_s:
        time.sleep(sleep_s)
    if frame is None:
        frame = pd.DataFrame()
    if len(frame) != expected:
        raise RuntimeError(
            f"{mode} query for {len(root_ids)} cells returned {len(frame)} rows "
            f"but the count said {expected} -- truncated or raced"
        )
    if frame.empty:
        return EMPTY.copy()
    return normalize_frame(frame, mode)
