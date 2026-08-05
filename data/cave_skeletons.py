"""Direct CAVE skeleton fetching with a local disk cache -- no segclr_db store.

segclr_db's Store/Writer/Database registry is not used anywhere in this
project. What IS reused, as a plain Python library (not a database), is:
  - segclr_db.cave.CAVESkeletonSource: the rate-limited generate/poll/download
    loop against CAVE's skeleton service. It "knows nothing about storage" by
    its own docstring -- built for exactly this kind of reuse -- and
    reimplementing CAVE's async-generation/readiness-polling/retry dance here
    would risk hammering a shared, rate-limited service that other labs use.
  - segclr_db.skeletons.normalize_cave_skeleton: pure translation of CAVE's
    wire-format dict into a Skeleton dataclass.
  - segclr_db.results.Skeleton: the dataclass itself.

Persistence here is one pickle file per root_id under CACHE_DIR, not Lance --
plain enough to inspect with `ls` and delete by hand if something needs redoing.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

_SEGCLR_DB_SRC = Path(__file__).resolve().parent.parent / "segclr_db" / "src"
if str(_SEGCLR_DB_SRC) not in sys.path:
    sys.path.insert(0, str(_SEGCLR_DB_SRC))

from segclr_db.cave import CAVEConfig, CAVESkeletonSource  # noqa: E402
from segclr_db.results import Skeleton  # noqa: E402
from segclr_db.skeletons import normalize_cave_skeleton  # noqa: E402

# CAVESkeletonSource.fetch(ids) is all-or-nothing: it returns only once
# generation + readiness-polling + download for the WHOLE list is done, so a
# 370-cell call can silently sit for up to readiness_timeout_s (1hr default)
# with nothing cached yet -- killing the job at any point during that call
# loses every skeleton in it, even ones CAVE already finished generating.
# Chunking the outer call and caching after each chunk bounds that loss to one
# chunk and gives a place to show real fetch progress instead of one big
# silent wait. Chunk size intentionally well under CAVE's own per-call caps
# (generation 10_000 / readiness 1_000 / download 500, see cave.py) -- this
# isn't working around CAVE's limits, just making our own progress visible.
DEFAULT_FETCH_CHUNK_SIZE = 40

CACHE_DIR = Path(__file__).resolve().parent / "skeleton_cache"

# Established by scripts/explore_cave_alignment.py: the public MICrONS release
# that the m343 label table (labeled_cell_m343_df_221011b.feather) is drawn
# from is CAVE datastack "minnie65_public". Materialization 343 itself is no
# longer in the datastack's live `get_versions()` list (materializations get
# retired), but skeleton fetching worked against it anyway -- skeletons key
# off root_id + skeleton_version, not off a still-valid materialization.
DATASTACK = "minnie65_public"
MAT_VERSION = 343


def default_cave_config(token: str) -> CAVEConfig:
    return CAVEConfig(datastack=DATASTACK, materialization_version=MAT_VERSION, token=token)


def _cache_path(root_id: int) -> Path:
    return CACHE_DIR / f"{root_id}.pkl"


def load_cached(root_id: int) -> Skeleton | None:
    path = _cache_path(root_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_skeletons(
    root_ids: list[int],
    cave_config: CAVEConfig,
    source: CAVESkeletonSource | None = None,
    log=print,
    chunk_size: int = DEFAULT_FETCH_CHUNK_SIZE,
    show_progress: bool = True,
) -> dict[int, Skeleton]:
    """Returns {root_id: Skeleton}, fetching from CAVE only for cache misses.

    Safe to re-run: cells already cached on disk are never re-fetched, so an
    interrupted run resumes rather than restarting -- same resumability
    principle as examples/ingest_skeletons.py, just without the Lance store.
    Fetches in chunks of `chunk_size` and caches each chunk to disk as soon as
    it returns (see DEFAULT_FETCH_CHUNK_SIZE for why), rather than one call
    covering every missing root_id.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source = source or CAVESkeletonSource(cave_config)

    result: dict[int, Skeleton] = {}
    missing = []
    for root_id in root_ids:
        cached = load_cached(root_id)
        if cached is not None:
            result[root_id] = cached
        else:
            missing.append(int(root_id))

    log(f"{len(root_ids)} requested, {len(result)} already cached, {len(missing)} to fetch")
    if not missing:
        return result

    refused = source.refusal_list()
    fetchable = [r for r in missing if r not in refused]
    if len(fetchable) < len(missing):
        log(f"  {len(missing) - len(fetchable)} root_ids on CAVE's refusal list, skipping")

    chunks = [fetchable[i : i + chunk_size] for i in range(0, len(fetchable), chunk_size)]
    iterator = chunks
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(chunks, desc="skeleton chunks", unit="chunk")

    n_ok, n_missing, n_bad = 0, 0, 0
    for chunk in iterator:
        raw, still_missing = source.fetch(chunk)
        for root_id, payload in raw.items():
            try:
                skel = normalize_cave_skeleton(root_id, payload, cave_config.skeleton_version)
            except (TypeError, ValueError) as e:
                log(f"  skeleton {root_id} failed normalization: {e}")
                n_bad += 1
                continue
            with open(_cache_path(root_id), "wb") as f:
                pickle.dump(skel, f)
            result[root_id] = skel
            n_ok += 1
        n_missing += len(still_missing)
        if show_progress:
            iterator.set_postfix(ok=n_ok, missing=n_missing, bad=n_bad)
        else:
            log(f"  chunk done: {len(raw)} fetched, {len(still_missing)} not ready (running total ok={n_ok})")

    if n_missing:
        log(f"  {n_missing} root_ids not returned by CAVE (not ready, or refused)")

    return result
