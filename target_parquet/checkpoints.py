"""Append-only checkpoint log pairing Singer state with written parquet files.

Used so a killed job can resume from the latest state whose parquet chunk(s)
actually made it to S3: walk the list and pick the last entry whose last_files
exist remotely. Append-only (not overwrite) so a crash between file write and
checkpoint write never advances state past durable files.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional

CHECKPOINTS_PATH = "checkpoints.json"


def _read_checkpoints(path: str = CHECKPOINTS_PATH) -> List[dict]:
    if not os.path.isfile(path):
        return []

    try:
        with open(path, "r") as f:
            content = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    return content if isinstance(content, list) else []


def append_checkpoint(
    state: Optional[Dict[str, Any]],
    last_files: Optional[Dict[str, str]] = None,
    path: str = CHECKPOINTS_PATH,
) -> None:
    """Append {state, last_files} for the latest closed parquet file(s)."""
    if not state:
        return

    entry_files = dict(last_files or {})
    checkpoints = _read_checkpoints(path)
    # STATE + end-of-pipe can both fire with the same pair; skip duplicates.
    if checkpoints:
        latest = checkpoints[-1]
        if latest.get("state") == state and latest.get("last_files") == entry_files:
            return

    checkpoints.append(
        {
            "state": copy.deepcopy(state),
            "last_files": entry_files,
        }
    )

    # Atomic replace so a crash mid-write does not leave a corrupt JSON file.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoints, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
