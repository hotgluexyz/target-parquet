"""Append-only checkpoint log pairing Singer state with written parquet files.

Used so a killed job can resume from the latest state whose parquet chunk(s)
actually made it to S3: walk the list and pick the last entry whose last_files
exist remotely. Append-only (not overwrite) so a crash between file write and
checkpoint write never advances state past durable files.

completed_streams lists streams the target processed whose bookmarks are
finalized (no starting_replication_value). Mid-sync that key is still present
while a stream is in progress; Singer removes it when the stream finishes.

all_data_processed is True only on the end-of-pipe checkpoint (all input
consumed, writers closed, final combine/rename done).

seq is a 0-based sequential number assigned when the entry is appended.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

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


def _bookmark_is_finished(bookmark: Optional[Dict[str, Any]]) -> bool:
    """True when the stream bookmark has no starting_replication_value.

    Singer sets starting_replication_value when a stream starts (value may be
    None for full-table) and removes it on finalize — finished full-table
    bookmarks are {}. If the key is still present (top-level or in any
    partition), the stream is not finished.
    """
    if bookmark is None:
        return False
    if "starting_replication_value" in bookmark:
        return False
    partitions = bookmark.get("partitions")
    if isinstance(partitions, list):
        for part in partitions:
            if isinstance(part, dict) and "starting_replication_value" in part:
                return False
    return True


def finished_completed_streams(
    state: Optional[Dict[str, Any]],
    processed_streams: Optional[Sequence[str]] = None,
) -> List[str]:
    """Processed streams whose bookmarks are finalized (no starting_replication_value).

    In-progress bookmarks keep starting_replication_value (including None for
    full-table). Finished full-table bookmarks are {}; incremental ones keep
    replication_key / replication_key_value without the starting marker.
    """
    bookmarks = (state or {}).get("bookmarks") or {}
    return sorted(
        {
            str(stream)
            for stream in (processed_streams or [])
            if _bookmark_is_finished(bookmarks.get(stream))
        }
    )


def _write_checkpoints(checkpoints: List[dict], path: str = CHECKPOINTS_PATH) -> None:
    logger.info(f"Writing checkpoints to {path}")
    # Atomic replace so a crash mid-write does not leave a corrupt JSON file.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoints, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def append_checkpoint(
    state: Optional[Dict[str, Any]],
    last_files: Optional[Dict[str, str]] = None,
    completed_streams: Optional[Sequence[str]] = None,
    all_data_processed: bool = False,
    path: str = CHECKPOINTS_PATH,
) -> None:
    """Append {seq, state, last_files, completed_streams} for the latest closed parquet file(s).

    When all_data_processed is True (end-of-pipe), the flag is set on the entry.
    If the latest checkpoint already matches state/files/streams, that entry is
    updated in place so the final flag is not lost to the duplicate skip.
    """
    if not state:
        return

    entry_files = dict(last_files or {})
    entry_completed = finished_completed_streams(state, completed_streams)
    checkpoints = _read_checkpoints(path)
    # STATE + end-of-pipe can both fire with the same pair; skip duplicates,
    # but promote the latest entry when this is the final checkpoint.
    if checkpoints:
        latest = checkpoints[-1]
        if (
            latest.get("state") == state
            and latest.get("last_files") == entry_files
            and sorted(latest.get("completed_streams") or []) == entry_completed
        ):
            if all_data_processed and not latest.get("all_data_processed"):
                latest["all_data_processed"] = True
                _write_checkpoints(checkpoints, path)
            return

    latest_seq = checkpoints[-1].get("seq") if checkpoints else None
    if isinstance(latest_seq, int):
        seq = latest_seq + 1
    else:
        seq = len(checkpoints)

    entry = {
        "seq": seq,
        "state": copy.deepcopy(state),
        "last_files": entry_files,
        "completed_streams": entry_completed,
    }
    if all_data_processed:
        entry["all_data_processed"] = True
    checkpoints.append(entry)
    _write_checkpoints(checkpoints, path)
