"""Append-only checkpoint log pairing Singer state with written parquet files.

Used so a killed job can resume from the latest state whose parquet chunk(s)
actually made it to S3: walk the list and pick the last entry whose last_files
exist remotely. Append-only (not overwrite) so a crash between file write and
checkpoint write never advances state past durable files.

completed_streams lists streams known finished (e.g. end-of-pipe). Mid-sync the
executor also infers completion from last_files history (sequential stream
changes), with parent/child interleaving treated as still active.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence

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
    completed_streams: Optional[Sequence[str]] = None,
    path: str = CHECKPOINTS_PATH,
) -> None:
    """Append {state, last_files, completed_streams} for the latest closed parquet file(s)."""
    if not state:
        return

    entry_files = dict(last_files or {})
    entry_completed = sorted({str(s) for s in (completed_streams or [])})
    checkpoints = _read_checkpoints(path)
    # STATE + end-of-pipe can both fire with the same pair; skip duplicates.
    if checkpoints:
        latest = checkpoints[-1]
        if (
            latest.get("state") == state
            and latest.get("last_files") == entry_files
            and sorted(latest.get("completed_streams") or []) == entry_completed
        ):
            return

    checkpoints.append(
        {
            "state": copy.deepcopy(state),
            "last_files": entry_files,
            "completed_streams": entry_completed,
        }
    )

    # Atomic replace so a crash mid-write does not leave a corrupt JSON file.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoints, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
