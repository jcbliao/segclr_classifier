"""Every checkpoint in results/ still rebuilds its model and loads its weights.

The architecture renames (fully_connected -> mpnn_complete, deepsets ->
pointwise_mlp) reached inside the checkpoints themselves: the stored
`ModelConfig`'s field names and `architecture` value, and the `state_dict` keys
that follow the renamed submodule. Both are rewritten by
`scripts/migrate_architecture_names.py`, and both fail quietly in the ways that
matter most -- a stale config aborts a `--resume` hours after submission, and a
stale state_dict key surfaces as a `load_state_dict` error at the very end of a
run, when the best weights are reloaded for the final evaluation.

So this rebuilds a `WindowClassifier` from each checkpoint's own stored config
and loads its weights strictly, which is exactly what `train_gnn.py` does on
resume and at final evaluation. CPU-only and reads nothing but the checkpoints.

    python scripts/check_migrated_checkpoints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "all_windows"


def main() -> int:
    hierarchy = load_hierarchy(load_manifest())

    paths = sorted(RESULTS.glob("*/checkpoint_*.pt"))
    if not paths:
        print("no checkpoints in results/all_windows/")
        return 0

    failures = []
    for path in paths:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            config = ckpt["config"]
            if not isinstance(config, ModelConfig):
                raise TypeError(f"stored config is {type(config).__name__}, not ModelConfig")
            # The equality the resume path performs. A field renamed on the
            # class but not in the pickle raises here rather than comparing
            # unequal, which is the failure this check exists for.
            if ModelConfig(**vars(config)) != config:
                raise ValueError("config does not compare equal to a rebuild of itself")
            model = WindowClassifier(config, hierarchy=hierarchy)
            model.load_state_dict(ckpt["model_state"])
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            failures.append((path, error))
            print(f"FAIL {path.parent.name}/{path.name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {path.parent.name}/{path.name}  ({config.architecture})")

    print(f"\n{len(paths) - len(failures)}/{len(paths)} checkpoint(s) rebuilt and loaded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
