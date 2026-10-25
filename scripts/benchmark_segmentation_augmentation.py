"""GPU-only synthetic benchmark of mask updates versus SegCLR inference.

No volume reads or output cache writes. Uses synchronized CUDA events, warmups,
and the same batch size, precision, layout and grouping as the generation runner.
"""

import argparse
import json
import statistics

import torch

from data.segmentation_augmentation import mask_variants, neighbor_step
from scripts.generate_cave_embedding_augmentations import CHECKPOINT
from segclr.utils.checkpointing import load_checkpoint_automatic


def measure(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return dict(
        median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--neighbors", type=int, choices=(6, 26), default=6)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--variants-per-forward", type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1 or a.radius < 1 or a.repeats < 1:
        p.error("batch size, radius and repeats must be positive")
    model = (
        load_checkpoint_automatic(CHECKPOINT, torch.device("cuda"))
        .eval()
        .to(memory_format=torch.channels_last_3d)
    )
    if a.compile_model:
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    side = 129 + 2 * a.radius
    mask = torch.zeros(
        (a.batch_size, 1, side, side, side), dtype=torch.bool, device="cuda"
    )
    mask[:, :, side // 2 - 5 : side // 2 + 5, side // 2 - 5 : side // 2 + 5, :] = True
    em = torch.rand((a.batch_size, 1, 129, 129, 129), device="cuda").to(
        memory_format=torch.channels_last_3d
    )
    dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[a.amp]

    def morphology(step):
        return list(mask_variants(mask, a.radius, a.neighbors, 129, step=step))

    def forward(batch):
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=dtype, enabled=dtype is not None),
        ):
            return model(batch, return_embeddings=True)

    compiled = torch.compile(neighbor_step, dynamic=False)
    result = dict(
        device=torch.cuda.get_device_name(),
        batch_size=a.batch_size,
        radius=a.radius,
        neighbors=a.neighbors,
        precision=a.amp,
        synthetic=True,
        inference=measure(lambda: forward(em), a.repeats),
        eager_morphology=measure(lambda: morphology(neighbor_step), a.repeats),
        compiled_morphology=measure(lambda: morphology(compiled), a.repeats),
    )
    eager = morphology(neighbor_step)
    fused = morphology(compiled)
    for (_, left), (_, right) in zip(eager, fused, strict=True):
        torch.testing.assert_close(left, right)

    def combined():
        for _, view in mask_variants(mask, a.radius, a.neighbors, 129, step=compiled):
            forward(torch.where(view, em, 0.0).to(memory_format=torch.channels_last_3d))

    result["combined_all_variants_serial"] = measure(combined, a.repeats)

    def grouped():
        group = []
        for index, (_, view) in enumerate(
            mask_variants(mask, a.radius, a.neighbors, 129, step=compiled)
        ):
            group.append(torch.where(view, em, 0.0))
            if len(group) == a.variants_per_forward or index == 2 * a.radius:
                forward(torch.cat(group).to(memory_format=torch.channels_last_3d))
                group = []

    result["combined_all_variants_grouped"] = measure(grouped, a.repeats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
