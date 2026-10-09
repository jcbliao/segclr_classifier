#!/usr/bin/env python3
"""Generate editable Neuroglancer JSON for one packed prediction model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_METADATA = Path(
    "/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/"
    "segclr_predictions/fold_test_validation/prediction_labels.json"
)
DEFAULT_SOURCE = (
    "precomputed://https://g-ffa18e.d1c26e.5898.data.globus.org/microns/"
    "segclr_predictions/fold_test_validation/skeletons"
)
SOURCE_NAMES = ("skeletons", "skeletons_l2")
DEFAULT_SEGMENT = "864691134886037498"
COMPONENTS = "xyzw"

# Deliberately plain RGB triples: the generated shader names every class, so
# users can edit these values without knowing which numeric class code is which.
COLORS = {
    "astrocyte": (0.12, 0.47, 0.71),
    "microglia": (1.00, 0.50, 0.05),
    "oligo": (0.17, 0.63, 0.17),
    "putative_cge": (0.84, 0.15, 0.16),
    "putative_parvalbumin": (0.58, 0.40, 0.74),
    "putative_somatostatin": (0.55, 0.34, 0.29),
    "pyramidal": (0.89, 0.47, 0.76),
    "thalamocortical": (0.50, 0.50, 0.50),
}


def glsl_name(label: str) -> str:
    return "".join(c.upper() if c.isalnum() else "_" for c in label)


def resolve_model(models: list[dict], query: str) -> dict:
    exact = [m for m in models if m["run"] == query]
    if exact:
        return exact[0]
    folded = query.casefold()
    matches = [m for m in models if folded in m["run"].casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"no model matches {query!r}; use --list-models")
    choices = "\n  ".join(m["run"] for m in matches)
    raise SystemExit(f"model query {query!r} is ambiguous:\n  {choices}")


def editable_constants(class_codes: dict[str, str], mode: str) -> list[str]:
    lines = [
        "// EDITABLE COLORS AND TYPE-SPECIFIC OPACITIES",
        "// Opacity is 1.0 by default. Skeleton edges expose RGB but not alpha,",
        "// so fractional values use deterministic screen-door transparency.",
    ]
    for code in sorted((int(k) for k in class_codes if int(k) >= 0)):
        label = class_codes[str(code)]
        name = glsl_name(label)
        rgb = COLORS.get(label, (0.75, 0.75, 0.75))
        lines.append(f"const vec3 COLOR_{name} = vec3({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}); // class {code}: {label}")
        lines.append(f"const float OPACITY_{name} = 1.0;")
    if mode == "correctness":
        lines.extend([
            "const vec3 COLOR_CORRECT = vec3(0.10, 0.80, 0.20);",
            "const vec3 COLOR_WRONG = vec3(0.95, 0.10, 0.15);",
        ])
    lines.append("const vec3 COLOR_UNAVAILABLE = vec3(0.25, 0.25, 0.25);")
    lines.append("const float OPACITY_UNAVAILABLE = 0.0;")
    return lines


def class_functions(class_codes: dict[str, str]) -> list[str]:
    color = ["vec3 predictionColor(float cls) {"]
    opacity = ["float predictionOpacity(float cls) {"]
    for code in sorted((int(k) for k in class_codes if int(k) >= 0)):
        name = glsl_name(class_codes[str(code)])
        color.append(f"  if (cls == {float(code):.1f}) return COLOR_{name};")
        opacity.append(f"  if (cls == {float(code):.1f}) return OPACITY_{name};")
    color.extend(["  return COLOR_UNAVAILABLE;", "}"])
    opacity.extend(["  return OPACITY_UNAVAILABLE;", "}"])
    return color + [""] + opacity


def model_index(metadata: dict, selected: dict) -> int:
    return next(i for i, model in enumerate(metadata["models"])
                if model["run"] == selected["run"])


def prediction_selector(metadata: dict) -> list[str]:
    lines = ["float predictionForModel(int selectedModel) {"]
    for index, model in enumerate(metadata["models"]):
        component = COMPONENTS[int(model["component"])]
        divisor = float(model["divisor"])
        lines.append(f"  // {index}: {model['run']}")
        lines.append(
            f"  if (selectedModel == {index}) return floor(mod(floor("
            f"prop_{model['vertex_attribute']}().{component} / {divisor:.1f}), 16.0)) - 1.0;"
        )
    lines.extend(["  return -1.0;", "}"])
    return lines


def make_shader(metadata: dict, model: dict, mode: str) -> str:
    selected_index = model_index(metadata, model)
    lines = [
        f"// SELECTED MODEL {selected_index}: {model['run']}",
        "// Change skeletonRendering.shaderControls.model in the layer JSON.",
        "// Model-index table is listed in predictionForModel below.",
        f"// MODE: {mode}",
        "// Decode: stored nibble 0 = unavailable; 1..8 = class codes 0..7.",
        f"#uicontrol int model slider(min=0, max={len(metadata['models']) - 1}, default={selected_index})",
        "",
        *editable_constants(metadata["class_codes"], mode),
        "",
        *class_functions(metadata["class_codes"]),
        "",
        *prediction_selector(metadata),
        "",
        "void main() {",
        "  float prediction = predictionForModel(model);",
        "  float typeOpacity = predictionOpacity(prediction);",
        "  if (typeOpacity <= 0.0) discard;",
    ]
    if mode == "rainbow":
        lines.append("  vec3 color = predictionColor(prediction);")
    else:
        lines.extend([
        "  float target = prop_target_class();",
            "  vec3 color = (prediction == target) ? COLOR_CORRECT : COLOR_WRONG;",
        ])
    lines.extend([
        "  // Screen-door opacity works for both line and point skeleton rendering.",
        "  float dither = fract(sin(dot(floor(gl_FragCoord.xy), vec2(12.9898, 78.233))) * 43758.5453);",
        "  if (dither > typeOpacity) discard;",
        "  emitRGB(color);",
        "}",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", help="exact run name or unique substring")
    ap.add_argument("--mode", choices=("correctness", "rainbow"), default="rainbow")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--skeleton-source", choices=SOURCE_NAMES,
                    help="select sibling database; overrides --source")
    ap.add_argument("--segment", action="append", default=[])
    ap.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    metadata = json.loads(args.metadata.read_text())
    models = metadata["models"]
    if args.list_models:
        print("\n".join(m["run"] for m in models))
        return
    if not args.model:
        ap.error("MODEL is required unless --list-models is used")
    if metadata.get("attribute_packing", {}).get("predictions_per_float") != 6:
        raise SystemExit(
            f"{args.metadata} is not a published 4-bit-packed export; wait for rebuild finalization"
        )
    model = resolve_model(models, args.model)
    shader = make_shader(metadata, model, args.mode)
    source = args.source
    if args.skeleton_source:
        source = DEFAULT_SOURCE.rsplit("/", 1)[0] + "/" + args.skeleton_source
    selected_index = model_index(metadata, model)
    layer = {
        "type": "segmentation",
        "name": f"{args.mode}__{model['run']}",
        "source": source,
        "segments": args.segment or [DEFAULT_SEGMENT],
        "skeletonRendering": {
            "shader": shader,
            "shaderControls": {"model": selected_index},
        },
        "selectedAlpha": 1,
        "notSelectedAlpha": 0,
    }
    rendered = json.dumps(layer, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
        print(args.output)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
