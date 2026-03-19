import json
from io import StringIO

import pyarrow.parquet as pq
import pytest

from target_parquet.target import TargetParquet
from target_parquet.writers import SingletonMeta, Writers


@pytest.fixture(autouse=True)
def isolate_test(tmp_path, monkeypatch):
    """Run each test in an isolated temp directory with a fresh Writers singleton."""
    SingletonMeta._instances.pop(Writers, None)
    Writers._writers = {}
    monkeypatch.chdir(tmp_path)
    yield
    if Writers in SingletonMeta._instances:
        try:
            SingletonMeta._instances[Writers].close_all()
        except Exception:
            pass
        SingletonMeta._instances.pop(Writers, None)
    Writers._writers = {}


def schema_message(stream, properties, key_properties=None):
    return json.dumps({
        "type": "SCHEMA",
        "stream": stream,
        "schema": {"type": "object", "properties": properties},
        "key_properties": key_properties or [],
    })


def record_message(stream, record):
    return json.dumps({
        "type": "RECORD",
        "stream": stream,
        "record": record,
    })


def state_message(value):
    return json.dumps({"type": "STATE", "value": value})


def run_target(messages, config=None):
    target = TargetParquet(config=config or {})
    target.listen(StringIO("\n".join(messages) + "\n"))
    return target


def read_parquet_for_stream(tmp_path, stream_name):
    files = sorted(tmp_path.glob(f"{stream_name}-*.parquet"))
    assert files, f"No parquet files found for stream '{stream_name}'"
    return pq.read_table(files[0])
