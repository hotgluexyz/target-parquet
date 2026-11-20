"""Tests for hybrid batch limits (record count and row payload bytes)."""

import json
import logging

from target_parquet.batch_limits import (
    DEFAULT_BATCH_MAX_BYTES,
    DEFAULT_BATCH_MAX_SIZE,
    resolve_batch_max_bytes,
    resolve_batch_max_size,
    row_payload_bytes,
)
from target_parquet.sinks import ParquetSink
from target_parquet.target import TargetParquet


def test_row_payload_bytes_matches_compact_json():
    row = {"id": "1", "name": "Sample", "qty": 2}
    expected = len(json.dumps(row, separators=(",", ":"), default=str).encode("utf-8"))
    assert row_payload_bytes(row) == expected


def test_resolve_defaults():
    logger = logging.getLogger("test")
    assert resolve_batch_max_size({}, logger) == DEFAULT_BATCH_MAX_SIZE
    assert resolve_batch_max_bytes({}, logger) == DEFAULT_BATCH_MAX_BYTES


def test_resolve_config_and_env(monkeypatch):
    logger = logging.getLogger("test")
    monkeypatch.setenv("PARQUET_BATCH_MAX_SIZE", "500")
    monkeypatch.setenv("PARQUET_BATCH_MAX_BYTES", "1000")
    assert resolve_batch_max_size({"BATCH_MAX_SIZE": 99}, logger) == 500
    assert resolve_batch_max_bytes({"BATCH_MAX_BYTES": 99}, logger) == 1000


def test_invalid_env_falls_back_to_config(monkeypatch):
    logger = logging.getLogger("test")
    monkeypatch.setenv("PARQUET_BATCH_MAX_SIZE", "not-a-number")
    assert resolve_batch_max_size({"BATCH_MAX_SIZE": 250}, logger) == 250


class TestParquetSinkIsFull:
    _SCHEMA = {
        "type": "object",
        "properties": {"id": {"type": ["string", "null"]}, "blob": {"type": ["string", "null"]}},
    }

    def _sink(self, config=None):
        target = TargetParquet(config=config or {})
        return ParquetSink(target, "items", self._SCHEMA, ["id"])

    def test_byte_limit_triggers_before_record_cap(self):
        sink = self._sink({"BATCH_MAX_SIZE": 10_000, "BATCH_MAX_BYTES": 50})
        ctx = {}
        sink.start_batch(ctx)
        sink._pending_batch = ctx
        fat = {"id": "1", "blob": "x" * 100}
        sink.tally_record_read()
        sink.process_record(fat, ctx)
        assert sink.is_full

    def test_record_cap_still_applies(self):
        sink = self._sink({"BATCH_MAX_SIZE": 2, "BATCH_MAX_BYTES": 0})
        ctx = {}
        sink.start_batch(ctx)
        sink._pending_batch = ctx
        for i in range(2):
            sink.tally_record_read()
            sink.process_record({"id": str(i), "blob": ""}, ctx)
        assert sink.is_full