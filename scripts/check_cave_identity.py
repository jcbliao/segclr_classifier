"""Who does the CAVE token in ~/.cloudvolume/secrets/*cave-secret.json actually
belong to? Uses caveclient.auth.AuthClient.get_tokens() (returns user_id --
deliberately NOT printed: that response also includes the raw token string,
which this script never touches) then get_user_information(user_id) for the
human-readable identity. Run via sbatch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db.cave import CAVEConfig  # noqa: E402


def main() -> int:
    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2

    # minnie65_public is the datastack we know this token authenticates
    # against successfully; auth identity is account-level, not
    # datastack-scoped, so any working datastack does for this lookup.
    config = CAVEConfig(datastack="minnie65_public", materialization_version=343, token=token)
    client = config.build_client()

    tokens = client.auth.get_tokens()
    user_ids = sorted({t["user_id"] for t in tokens})
    print(f"token belongs to user_id(s): {user_ids}  ({len(tokens)} token(s) on this account)")

    info = client.auth.get_user_information(user_ids)
    for entry in info:
        safe = {k: v for k, v in entry.items() if k.lower() not in ("token",)}
        print(safe)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
