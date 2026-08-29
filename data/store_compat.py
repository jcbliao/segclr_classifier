"""Read a schema-v4 store without disturbing the vendored v3 clone.

The shared store at STORE_ROOT is schema v4; `segclr_db/` in this repo is
pinned at the commit that writes v3, and `store.open_store` refuses a version
mismatch in both directions by design. The clone is also installed **editable**
into `~/.conda/envs/segclr`, which is what the embedding pipeline imports, so
pulling it moves that pipeline at the same instant -- a decision that belongs to
whoever is running it, not to a read-only analysis job.

So this points the import at a separate checkout instead. The subtlety is that
the editable install is a PEP 660 **`sys.meta_path` finder**, not a path entry:
it resolves `segclr_db` before `sys.path` is ever consulted, so prepending to
`sys.path` or setting PYTHONPATH does nothing. The finder has to be removed.

**This is a shim, not the fix.** The fix is `git -C segclr_db pull` (a
fast-forward to a7e5168, which is SCHEMA_VERSION 4), taken deliberately and at a
moment when moving the embedding pipeline is acceptable. Delete this module then.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

#: Checkout of segclr_db at a commit that reads schema v4.
DEFAULT_V4_CHECKOUT = Path("/orcd/scratch/orcd/013/jcbliao/segclr_db_v4")

EXPECTED_SCHEMA_VERSION = 4


def use_v4(checkout: Path | str = DEFAULT_V4_CHECKOUT) -> Path:
    """Make `import segclr_db` resolve to `checkout/src`. Verifies the version.

    Fails loudly if the wrong generation ends up imported -- reading a v4 store
    through v3 code is exactly the silent mixing the version guard exists to
    stop, and a shim that half-works would reintroduce it.
    """
    checkout = Path(checkout)
    src = checkout / "src"
    if not (src / "schema.py").exists():
        raise FileNotFoundError(f"no segclr_db source at {src}")

    # 1. drop the editable finder that would otherwise win
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if "__editable___segclr_db" not in getattr(f, "__module__", "")
        and "__editable___segclr_db" not in type(f).__module__
    ]
    # 2. drop anything already imported from the v3 tree
    for name in [m for m in sys.modules if m == "segclr_db" or m.startswith("segclr_db.")]:
        del sys.modules[name]
    # 3. a package root whose only entry is segclr_db -> checkout/src
    pkgroot = checkout / "_pkgroot"
    pkgroot.mkdir(exist_ok=True)
    link = pkgroot / "segclr_db"
    if not link.exists():
        link.symlink_to(src)
    sys.path.insert(0, str(pkgroot))

    schema = importlib.import_module("segclr_db.schema")
    got = getattr(schema, "SCHEMA_VERSION", None)
    if got != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"expected SCHEMA_VERSION {EXPECTED_SCHEMA_VERSION} from {src}, got {got} "
            f"(resolved to {schema.__file__})"
        )
    return checkout
