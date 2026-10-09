#!/usr/bin/env python3
"""Build a compact Spelunker link for one model and ground-truth cell type."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

from make_neuroglancer_prediction_shader import (
    DEFAULT_METADATA,
    DEFAULT_SOURCE,
    SOURCE_NAMES,
    make_shader,
    model_index,
    resolve_model,
)

VIEWER = "https://spelunker.cave-explorer.org/#!"
DEFAULT_CACHE = Path("/orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache")
CELL_TYPES = (
    "astrocyte", "microglia", "oligo", "putative_cge",
    "putative_parvalbumin", "putative_somatostatin", "pyramidal",
    "thalamocortical",
)

IMAGE_SHADER = """#uicontrol float black slider(min=0, max=1, default=0.0)
#uicontrol float white slider(min=0, max=1, default=1.0)
float rescale(float value) {
  return (value - black) / (white - black);
}
void main() {
  float val = toNormalized(getDataValue());
  if (val < black) {
    emitRGB(vec3(0, 0, 0));
  } else if (val > white) {
    emitRGB(vec3(1.0, 1.0, 1.0));
  } else {
    emitGrayscale(rescale(val));
  }
}
"""


def roots_for_type(metadata: dict, model: dict, cell_type: str,
                   cache_dir: Path) -> list[str]:
    inverse = {label: int(code) for code, label in metadata["class_codes"].items()}
    code = inverse[cell_type]
    cache = cache_dir / f"{model['run']}.npz"
    if not cache.is_file():
        raise SystemExit(f"missing prediction cache: {cache}")
    with np.load(cache) as arrays:
        split = arrays["split"].astype(str)
        roots = arrays["root_id"].astype(np.uint64)
        targets = arrays["target"].astype(np.int16)
    selected = np.unique(roots[(split == "test") & (targets == code)])
    if not len(selected):
        raise SystemExit(f"no held-out roots found for {cell_type}")
    return [str(root) for root in selected]


def prediction_layer(source: str, model: dict, mode: str, cell_type: str,
                     roots: list[str], shader: str, selected_index: int) -> dict:
    layer_name = f"{cell_type}__{mode}__{model['run']}"
    return {
        "type": "segmentation", "source": source, "segments": roots,
        "name": layer_name,
        "skeletonRendering": {
            "shader": shader,
            "shaderControls": {"model": selected_index},
        },
        "selectedAlpha": 1, "notSelectedAlpha": 0, "tab": "rendering",
    }


def viewer_state(source: str, model: dict, modes: list[str], cell_type: str,
                 roots: list[str], shaders: dict[str, str], selected_index: int) -> dict:
    prediction_layers = [
        prediction_layer(source, model, mode, cell_type, roots, shaders[mode], selected_index)
        for mode in modes
    ]
    selected_name = prediction_layers[0]["name"]
    return {
        "dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [4e-8, "m"]},
        "position": [122597, 95833.5, 21354],
        "crossSectionScale": 0.25,
        "projectionOrientation": [
            -0.756763219833374, 0.17967496812343597,
            0.10323526710271835, 0.6199748516082764,
        ],
        "projectionScale": 669795.669981548,
        "layers": [
            {
                "type": "image",
                "source": (
                    "precomputed://https://bossdb-open-data.s3.amazonaws.com/"
                    "iarpa_microns/minnie/minnie65/em"
                ),
                "tab": "source", "shader": IMAGE_SHADER, "name": "img",
            },
            {
                "type": "segmentation",
                "source": [
                    {
                        "url": (
                            "graphene://middleauth+https://minnie.microns-daf.com/"
                            "segmentation/table/minnie3_v1"
                        ),
                        "subsources": {
                            "default": True, "graph": True, "bounds": True, "mesh": True,
                        },
                        "enableDefaultSubsources": False,
                    },
                    {
                        "url": (
                            "precomputed://middleauth+https://minnie.microns-daf.com/"
                            "skeletoncache/api/v1/minnie65_phase3_v1/precomputed/skeleton"
                        ),
                        "subsources": {"default": True},
                        "enableDefaultSubsources": False,
                    },
                ],
                "tab": "source", "segments": [], "name": "seg",
            },
            *prediction_layers,
        ],
        "selectedLayer": {"layer": selected_name, "visible": True},
        "showSlices": False,
        "layout": "xy-3d",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", help="exact run name or unique substring")
    ap.add_argument("--mode", choices=("correctness", "rainbow", "both"),
                    default="rainbow")
    ap.add_argument("--type", choices=CELL_TYPES,
                    help="ground-truth type whose held-out neurons are selected")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--skeleton-source", choices=SOURCE_NAMES, default="skeletons",
                    help="prediction database under the published fold directory")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--output", type=Path, help="write the URL here instead of stdout")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.skeleton_source == "skeletons_l2" and args.metadata == DEFAULT_METADATA:
        args.metadata = DEFAULT_METADATA.with_name("prediction_labels_l2.json")
    metadata = json.loads(args.metadata.read_text())
    models = metadata["models"]
    if args.list_models:
        print("\n".join(m["run"] for m in models))
        return
    if not args.model:
        ap.error("MODEL is required unless --list-models is used")
    if not args.type:
        ap.error("--type is required")
    if metadata.get("attribute_packing", {}).get("predictions_per_float") != 6:
        raise SystemExit(
            f"{args.metadata} is not a published 4-bit-packed export; wait for rebuild finalization"
        )
    model = resolve_model(models, args.model)
    selected_index = model_index(metadata, model)
    if args.source == DEFAULT_SOURCE:
        args.source = DEFAULT_SOURCE.rsplit("/", 1)[0] + "/" + args.skeleton_source
    roots = roots_for_type(metadata, model, args.type, args.cache_dir)
    modes = ["rainbow", "correctness"] if args.mode == "both" else [args.mode]
    shaders = {mode: make_shader(metadata, model, mode) for mode in modes}
    compact = json.dumps(
        viewer_state(args.source, model, modes, args.type, roots, shaders, selected_index),
        separators=(",", ":"),
    )
    url = VIEWER + quote(compact, safe="")
    if args.output:
        args.output.write_text(url + "\n")
        print(
            f"{args.output} ({len(roots)} {args.type} roots, "
            f"{len(modes)} prediction layer(s), {len(url):,} URL characters)"
        )
    else:
        sys.stdout.write(url + "\n")


if __name__ == "__main__":
    main()
