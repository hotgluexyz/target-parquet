"""In-memory batch limits for ParquetSink (record count and estimated payload bytes)."""

import json
import logging
import os
from typing import Any, Mapping

DEFAULT_BATCH_MAX_SIZE = 10_000
DEFAULT_BATCH_MAX_BYTES = 64 * 1024 * 1024

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


def row_payload_bytes(row: dict) -> int:
    """Return UTF-8 byte length of a compact JSON encoding of the row (batch weight proxy)."""
    if _orjson is not None:
        return len(_orjson.dumps(row, option=_orjson.OPT_NON_STR_KEYS))
    payload = json.dumps(row, separators=(",", ":"), default=str).encode("utf-8")
    return len(payload)


def resolve_batch_max_size(config: Mapping[str, Any], logger: logging.Logger) -> int:
    """Record cap per drain from env or config (default 10k)."""
    env_value = os.getenv("PARQUET_BATCH_MAX_SIZE")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            logger.warning(
                "Invalid PARQUET_BATCH_MAX_SIZE %r; using config/default",
                env_value,
            )
    raw = config.get("BATCH_MAX_SIZE")
    if raw is not None:
        return int(raw)
    return DEFAULT_BATCH_MAX_SIZE


def resolve_batch_max_bytes(config: Mapping[str, Any], logger: logging.Logger) -> int:
    """Payload byte cap per drain from env or config (default 64 MiB; 0 disables)."""
    env_value = os.getenv("PARQUET_BATCH_MAX_BYTES")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            logger.warning(
                "Invalid PARQUET_BATCH_MAX_BYTES %r; using config/default",
                env_value,
            )
    raw = config.get("BATCH_MAX_BYTES")
    if raw is not None:
        return int(raw)
    return DEFAULT_BATCH_MAX_BYTES