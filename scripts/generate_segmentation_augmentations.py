"""Prepare, benchmark, generate and summarize CAVE mask-augmented embeddings.

No jobs are submitted. `prepare` reads existing classifier geometry only;
`infer` is an explicit GPU operation using the production double-buffered reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/presynaptic/segmentation_augmentation"
CLASSIFIER_RECORD = ROOT / (
    "results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold0/pre_post/"
    "gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold0.json"
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_hash(ids, xyz):
    return hashlib.sha256(ids.tobytes() + xyz.tobytes()).hexdigest()


def select_roots(cohort, limit, stratified=False, include_held_out=False):
    roots = sorted(
        int(root)
        for root, item in cohort["cells"].items()
        if include_held_out or item["split"] == "train"
    )
    if not limit or limit >= len(roots):
        return roots
    if not stratified:
        return roots[:limit]
    from collections import defaultdict

    groups = defaultdict(list)
    for root in roots:
        groups[cohort["cells"][str(root)]["cell_type"]].append(root)
    rng = np.random.default_rng(20261016)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < limit:
        for label in sorted(groups):
            if groups[label]:
                chosen.append(groups[label].pop())
                if len(chosen) == limit:
                    break
    return sorted(chosen)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()) + "\n")
    os.replace(tmp, path)


def prepare(args):
    from scripts.generate_cave_embedding_augmentations import (
        CHECKPOINT,
        target_nodes,
    )

    classifier_args = json.loads(CLASSIFIER_RECORD.read_text())["args"]
    manifest = args.manifest or Path(classifier_args["manifest"])
    database = args.database or Path(classifier_args["presynaptic_database"])
    cohort = json.loads(manifest.read_text())
    roots = select_roots(
        cohort, args.limit_cells, args.stratify_cells, args.include_held_out
    )
    items, missing = [], []
    for root_index, root in enumerate(roots, 1):
        source = database / "cells" / f"{root}.npz"
        if not source.exists():
            missing.append(root)
            continue
        ids, xyz = target_nodes(source)
        if args.nodes_per_cell:
            chosen = (
                np.linspace(
                    0, len(ids) - 1, min(args.nodes_per_cell, len(ids)), dtype=int
                )
                if len(ids)
                else []
            )
            ids, xyz = ids[chosen], xyz[chosen]
        items.append(
            dict(
                root_id=root,
                source=str(source.resolve()),
                n_nodes=len(ids),
                geometry_sha256=geometry_hash(ids, xyz),
                split=cohort["cells"][str(root)]["split"],
            )
        )
        if root_index % 50 == 0:
            print(f"Prepared geometry for {root_index}/{len(roots)} cells", flush=True)
    if missing:
        raise ValueError(
            f"{len(missing)} source cells missing; first IDs: {missing[:10]}"
        )
    shards = [[] for _ in range(args.tasks)]
    loads = [0] * args.tasks
    for item in sorted(items, key=lambda item: -item["n_nodes"]):
        rank = int(np.argmin(loads))
        shards[rank].append(item)
        loads[rank] += item["n_nodes"]
    plan = dict(
        format_version=1,
        manifest=str(manifest.resolve()),
        manifest_sha256=sha(manifest),
        classifier_record=str(CLASSIFIER_RECORD.resolve()),
        classifier_record_sha256=sha(CLASSIFIER_RECORD),
        checkpoint=str(CHECKPOINT.resolve()),
        checkpoint_sha256=sha(CHECKPOINT),
        radius=args.radius,
        neighbors=args.neighbors,
        crop_size=129,
        resolution_nm=[32, 32, 40],
        materialization=1718,
        node_set_id="cave_skeleton_v4",
        selection="existing CAVE presynaptic conf0.7 fold0 target_nodes; 5um match cutoff",
        include_held_out=args.include_held_out,
        nodes_per_cell=args.nodes_per_cell,
        cell_selection="balanced by cell type, seed 20261016"
        if args.stratify_cells
        else "sorted root IDs",
        shards=shards,
        node_loads=loads,
        total_nodes=sum(loads),
        cells=len(items),
        precision=args.amp,
        variants_per_forward=args.variants_per_forward,
        output=str(args.output.resolve()),
    )
    if args.plan.exists():
        raise ValueError("plan already exists; choose a new plan filename")
    atomic_json(args.plan, plan)
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "cells",
                    "total_nodes",
                    "node_loads",
                    "radius",
                    "neighbors",
                    "output",
                )
            },
            indent=2,
        )
    )


def inference_class():
    import torch
    from segclr.data.crops import CropLoader
    from scripts.generate_cave_embedding_augmentations import FastAugInference
    from scripts.generate_all_cave_augmentations import MultiAugInference
    from data.segmentation_augmentation import (
        mask_variants,
        variant_names,
        neighbor_step,
    )

    class HaloCrops:
        def __init__(self, base, radius):
            self.base, self.radius = base, radius
            self.box_size = base.box_size
            self.halo = CropLoader(
                em_path=base.em_path,
                seg_path=base.seg_path,
                scale=base.scale,
                seg_scale=base.seg_scale,
                box_size=self.box_size + 2 * radius,
                oob="pad",
            )

        def nm_to_voxel(self, xyz):
            return self.halo.nm_to_voxel(xyz)

        def fetch_crop(self, center, root):
            em = self.base.load_em_block(center)
            mask = self.halo.load_mask_block(center, root)
            size = self.halo.box_size
            start = np.asarray(center) - size // 2
            domain = self.halo.seg.domain
            axes = [
                (
                    (np.arange(size) + start[i] >= domain[i].inclusive_min)
                    & (np.arange(size) + start[i] < domain[i].exclusive_max)
                )
                for i in range(3)
            ]
            valid = (
                axes[2][:, None, None] & axes[1][None, :, None] & axes[0][None, None, :]
            )
            return em, torch.from_numpy(
                np.concatenate([mask.numpy(), valid[None]], axis=0)
            )

    class SegmentationInference(MultiAugInference):
        def __init__(self, plan, compile_morphology=False, **kwargs):
            super().__init__("gray", 0, 0, amp=plan["precision"], **kwargs)
            self.radius, self.neighbors = plan["radius"], plan["neighbors"]
            self.names = variant_names(self.radius)
            self.crops = HaloCrops(self.crops, self.radius)
            # Open volume handles before reader threads see the lazy properties.
            self.crops.base.em
            self.crops.halo.seg
            if not np.array_equal(self.crops.halo.resolution, plan["resolution_nm"]):
                raise ValueError("crop resolution differs from the frozen plan")
            self.step = (
                torch.compile(neighbor_step, dynamic=False)
                if compile_morphology
                else neighbor_step
            )
            self.timings = []

        def _ensure_buffers(self):
            if self._buffers is not None:
                return
            box, halo = 129, 129 + 2 * self.radius
            self._buffers = [
                (
                    torch.empty(
                        (self.batch_size, 1, box, box, box), dtype=torch.uint8
                    ).pin_memory(),
                    torch.empty(
                        (self.batch_size, 2, halo, halo, halo), dtype=torch.bool
                    ).pin_memory(),
                )
                for _ in range(2)
            ]
            self._buffer_views = [
                (em.numpy(), mask.numpy()) for em, mask in self._buffers
            ]

        def embed(self, root, xyz, ids):
            # Use the ordinary SegCLR ordering/failure checks, not stochastic policies.
            return FastAugInference.embed(self, root, xyz, ids)

        def _forward(self, slot, n_ok):
            events = []

            def begin(kind):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                events.append((kind, start, end))
                return end

            end = begin("transfer_normalize")
            em_buf, mask_buf = self._buffers[slot]
            em = em_buf[:n_ok].to(self.device, non_blocking=True).float().div_(255)
            masks = mask_buf[:n_ok].to(self.device, non_blocking=True)
            end.record()
            valid = masks[:, 1:2]
            iterator = iter(
                mask_variants(
                    masks[:, :1], self.radius, self.neighbors, 129, valid, self.step
                )
            )
            embeddings, counts, group = [], [], []
            for index in range(len(self.names)):
                end = begin("morphology")
                name, mask = next(iterator)
                end.record()
                end = begin("mask_and_count")
                counts.append(mask.sum(dim=(1, 2, 3, 4)))
                batch = torch.where(mask, em, 0.0)
                if n_ok < self.batch_size:
                    batch = torch.cat(
                        [
                            batch,
                            batch.new_zeros((self.batch_size - n_ok, *batch.shape[1:])),
                        ]
                    )
                group.append(batch)
                end.record()
                if (
                    len(group) == self.variants_per_forward
                    or index == len(self.names) - 1
                ):
                    batch = torch.cat(group).to(memory_format=torch.channels_last_3d)
                    end = begin("model")
                    with (
                        torch.inference_mode(),
                        torch.autocast(
                            "cuda",
                            dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None,
                        ),
                    ):
                        _, vectors = self.model(batch, return_embeddings=True)
                    end.record()
                    embeddings.extend(
                        part[:n_ok].float() for part in vectors.split(self.batch_size)
                    )
                    group = []
            # Counts stay aligned with embeddings through inherited Morton-order reversal.
            end = begin("download")
            final = (
                torch.cat(
                    [
                        *embeddings,
                        torch.stack(counts, dim=1).float(),
                        (~valid).flatten(1).any(1).float()[:, None],
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            end.record()
            end.synchronize()
            record = {"n_crops": n_ok}
            for kind, start, stop in events:
                record[kind + "_ms"] = record.get(
                    kind + "_ms", 0.0
                ) + start.elapsed_time(stop)
            self.timings.append(record)
            return final

    return SegmentationInference


def selected_nodes(item, plan):
    from scripts.generate_cave_embedding_augmentations import target_nodes

    ids, xyz = target_nodes(Path(item["source"]))
    if plan["nodes_per_cell"]:
        chosen = (
            np.linspace(
                0, len(ids) - 1, min(plan["nodes_per_cell"], len(ids)), dtype=int
            )
            if len(ids)
            else []
        )
        ids, xyz = ids[chosen], xyz[chosen]
    if geometry_hash(ids, xyz) != item["geometry_sha256"]:
        raise ValueError(f"source geometry changed: {item['root_id']}")
    return ids, xyz


def publish_cell(table, dest, timing_path, timing):
    import pyarrow.parquet as pq

    started = time.perf_counter()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".tmp.{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, dest)
    timing["write_seconds"] = time.perf_counter() - started
    atomic_json(timing_path, timing)


def infer(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data.segmentation_augmentation import variant_names

    plan = json.loads(args.plan.read_text())
    digest = sha(args.plan)
    from scripts.generate_cave_embedding_augmentations import CHECKPOINT

    if sha(CHECKPOINT) != plan["checkpoint_sha256"]:
        raise ValueError("inference checkpoint differs from frozen plan")
    if sha(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("classifier manifest changed")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"plan_{digest}.json", plan)
    names = variant_names(plan["radius"])
    runner = inference_class()(
        plan,
        batch_size=args.batch_size,
        num_threads=args.workers,
        variants_per_forward=plan["variants_per_forward"],
        compile_model=args.compile,
        compile_morphology=args.compile_morphology,
    )
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as writers:
        pending = []
        for item in plan["shards"][args.task_id]:
            if len(pending) >= 2:
                pending.pop(0).result()
            root = item["root_id"]
            dest = output / "parts" / f"{root}.parquet"
            ids, xyz = selected_nodes(item, plan)
            if dest.exists():
                file = pq.ParquetFile(dest)
                if (file.schema_arrow.metadata or {}).get(
                    b"plan_sha256"
                ) != digest.encode() or file.metadata.num_rows != len(ids) * len(names):
                    raise ValueError(
                        f"existing output belongs to another plan or is incomplete: {dest}"
                    )
                continue
            started = time.perf_counter()
            got, packed = runner.embed(root, xyz, np.arange(len(ids), dtype=np.int32))
            np.testing.assert_array_equal(got, np.arange(len(ids)))
            n, a = len(ids), len(names)
            vectors = packed[:, : a * 64].reshape(n, a, 64)
            counts = (packed[:, a * 64 : a * 65].astype(np.int32)
                      if n else np.empty((0,a),np.int32))
            if not np.isfinite(vectors).all():
                raise ValueError(f"nonfinite embeddings: {root}")
            baseline = np.repeat(counts[:, 0], a)
            augmented = counts.reshape(-1)
            table = pa.table(
                dict(
                    root_id=np.full(n * a, root, np.int64),
                    node_id=np.repeat(ids, a),
                    augmentation_id=np.tile(
                        [f"segmask_n{plan['neighbors']}_{name}" for name in names], n
                    ),
                    augmentation_sample=np.zeros(n * a, np.int32),
                    embedding=pa.FixedSizeListArray.from_arrays(
                        pa.array(vectors.reshape(-1)), 64
                    ),
                    baseline_voxels=baseline,
                    mask_voxels=augmented,
                    delta_voxels=augmented - baseline,
                    halo_clipped=np.repeat(packed[:, -1].astype(bool), a),
                )
            )
            table = table.replace_schema_metadata({b"plan_sha256": digest.encode()})
            pending.append(
                writers.submit(
                    publish_cell,
                    table,
                    dest,
                    output / "timings" / f"{root}.json",
                    dict(
                        root_id=root,
                        seconds=time.perf_counter() - started,
                        batches=runner.timings,
                    ),
                )
            )
            runner.timings = []
            print(f"{root}: queued {n} crops x {a} variants", flush=True)
        for future in pending:
            future.result()


def summarize(args):
    import duckdb

    plan = json.loads(args.plan.read_text())
    output = Path(plan["output"])
    files = list((output / "parts").glob("*.parquet"))
    if not files:
        raise ValueError("no embedding parts yet")
    import pyarrow.parquet as pq

    digest = sha(args.plan).encode()
    expected = {item["root_id"]: item for shard in plan["shards"] for item in shard}
    for path in files:
        if (
            int(path.stem) not in expected
            or (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"plan_sha256")
            != digest
        ):
            raise ValueError(f"unexpected output or plan hash: {path}")
    db = duckdb.connect(str(output / "embeddings.duckdb"))
    db.execute("SET memory_limit='24GB'")
    db.execute("SET threads=4")
    db.execute("SET temp_directory=?", [str(output / "summary_tmp")])
    db.read_parquet([str(path.resolve()) for path in files]).create_view(
        "embeddings", replace=True
    )
    stats = db.sql("""SELECT augmentation_id, count(*) AS n, count(DISTINCT root_id) AS cells,
        avg(mask_voxels) AS mean_voxels, quantile_cont(mask_voxels,[0.01,0.05,0.25,0.5,0.75,0.95,0.99]) AS voxel_quantiles,
        avg(delta_voxels) AS mean_delta_voxels,
        quantile_cont(delta_voxels,[0.05,0.5,0.95]) AS delta_quantiles,
        avg(CASE WHEN baseline_voxels>0 THEN CAST(delta_voxels AS DOUBLE)/baseline_voxels END) AS mean_fractional_change,
        quantile_cont(CASE WHEN baseline_voxels>0 THEN CAST(delta_voxels AS DOUBLE)/baseline_voxels END,[0.05,0.5,0.95]) AS fractional_change_quantiles,
        count(*) FILTER (WHERE mask_voxels=0) AS empty_masks,
        count(*) FILTER (WHERE baseline_voxels=0) AS baseline_empty_masks,
        count(*) FILTER (WHERE halo_clipped) AS halo_clipped_masks,
        avg(mask_voxels/2146689.0) AS mean_occupancy,
        avg(mask_voxels*0.00004096) AS mean_volume_um3,
        avg(list_cosine_similarity(embedding,baseline_embedding)) AS mean_baseline_cosine,
        quantile_cont(list_cosine_similarity(embedding,baseline_embedding),[0.05,0.5,0.95]) AS baseline_cosine_quantiles
        FROM (SELECT e.*,b.embedding AS baseline_embedding FROM embeddings e
              JOIN embeddings b USING(root_id,node_id) WHERE b.augmentation_id LIKE '%_clean')
        GROUP BY augmentation_id ORDER BY augmentation_id""").to_df()
    stats.to_csv(output / "mask_statistics.csv", index=False)
    # Histogram bins provide distributions without materializing all masks in RAM.
    db.sql("""SELECT augmentation_id, floor(mask_voxels/1000)*1000 AS voxel_bin,
              count(*) AS n FROM embeddings GROUP BY ALL ORDER BY augmentation_id,voxel_bin""").write_parquet(
        str(output / "mask_histograms.parquet")
    )
    db.sql("""SELECT augmentation_id,
        floor(100.0*delta_voxels/nullif(baseline_voxels,0)) AS percent_change_bin,
        count(*) AS n FROM embeddings GROUP BY ALL ORDER BY augmentation_id,percent_change_bin""").write_parquet(
        str(output / "change_histograms.parquet")
    )
    db.sql("""SELECT root_id,augmentation_id,count(*) AS n,avg(mask_voxels) AS mean_voxels,
        avg(delta_voxels) AS mean_delta_voxels,count(*) FILTER (WHERE mask_voxels=0) AS empty_masks
        FROM embeddings GROUP BY ALL ORDER BY root_id,augmentation_id""").write_csv(
        str(output / "per_cell_statistics.csv"), header=True
    )
    timings = [
        json.loads(path.read_text()) for path in (output / "timings").glob("*.json")
    ]
    timing_totals = {}
    for record in timings:
        for batch in record["batches"]:
            for name, value in batch.items():
                timing_totals[name] = timing_totals.get(name, 0) + value
    if timing_totals.get("model_ms"):
        timing_totals["morphology_over_model_time"] = (
            timing_totals.get("morphology_ms", 0) / timing_totals["model_ms"]
        )
    atomic_json(
        output / "summary.json",
        dict(
            completed_cells=len(files),
            expected_cells=len(expected),
            complete=len(files) == len(expected),
            statistics=stats.to_dict(orient="records"),
            timing_totals=timing_totals,
        ),
    )
    db.close()
    print(stats.to_string(index=False))


def pack(args):
    """Export choices in the current presynaptic dataset's existing format."""
    import pyarrow.parquet as pq
    from data.segmentation_augmentation import variant_names
    from scripts.pack_cave_training_augmentations import atomic_save

    plan = json.loads(args.plan.read_text())
    output = Path(plan["output"])
    names = variant_names(plan["radius"])
    a = len(names)
    for shard in plan["shards"]:
        for item in shard:
            root = item["root_id"]
            path = output / "parts" / f"{root}.parquet"
            table = pq.read_table(path)
            if (table.schema.metadata or {}).get(b"plan_sha256") != sha(
                args.plan
            ).encode():
                raise ValueError(f"plan mismatch: {path}")
            n = item["n_nodes"]
            if len(table) != n * a:
                raise ValueError(f"incomplete embedding part: {path}")
            ids = np.asarray(table["node_id"]).reshape(n, a)
            if not np.all(ids == ids[:, :1]) or len(np.unique(ids[:, 0])) != n:
                raise ValueError(f"invalid node ordering: {path}")
            expected = np.tile(
                [f"segmask_n{plan['neighbors']}_{name}" for name in names], n
            )
            np.testing.assert_array_equal(
                np.asarray(table["augmentation_id"]), expected
            )
            values = np.asarray(table["embedding"].combine_chunks().values).reshape(
                n, a, 64
            )
            atomic_save(
                output / "training_choices" / "segmask" / "cells" / f"{root}.npz",
                root_id=np.asarray(root, np.uint64),
                node_ids=ids[:, 0],
                embedding_choices=values,
                augmentation_id=np.asarray("segmask"),
                choice_labels=np.asarray(names),
                plan_sha256=np.asarray(sha(args.plan)),
            )
    atomic_json(
        output / "training_choices" / "metadata.json",
        dict(
            policy="segmask",
            choice_labels=names,
            plan_sha256=sha(args.plan),
            pilot=bool(plan["nodes_per_cell"]),
            source=str(output),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer", "summarize", "pack"))
    parser.add_argument("--plan", type=Path, default=OUTPUT / "pilot_plan.json")
    parser.add_argument("--output", type=Path, default=OUTPUT / "pilot")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--radius", type=int, choices=range(0, 9), default=3)
    parser.add_argument("--neighbors", type=int, choices=(6, 26), default=6)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--limit-cells", type=int, default=3)
    parser.add_argument("--stratify-cells", action="store_true")
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--nodes-per-cell", type=int, default=128)
    parser.add_argument(
        "--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variants-per-forward", type=int, default=4)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-morphology", action="store_true")
    args = parser.parse_args()
    if (
        args.tasks < 1
        or args.batch_size < 1
        or args.workers < 1
        or args.variants_per_forward < 1
        or args.limit_cells < 0
        or args.nodes_per_cell < 0
    ):
        parser.error(
            "counts must be positive; use zero cells/nodes limits for the full cohort"
        )
    {"prepare": prepare, "infer": infer, "summarize": summarize, "pack": pack}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
