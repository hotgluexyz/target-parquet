import gc
import json
from io import StringIO

import pyarrow.parquet as pq
import pytest

from target_parquet.target import TargetParquet
from target_parquet.writers import SingletonMeta, Writers


def _reset_writers():
    """Close every ParquetWriter and drop the Writers singleton."""
    instance = SingletonMeta._instances.pop(Writers, None)
    if instance is not None:
        instance.close_all()
    for writer in list(Writers._writers.values()):
        if writer.is_open:
            writer.close()
    Writers._writers.clear()


@pytest.fixture(autouse=True)
def isolate_test(tmp_path, monkeypatch):
    """Run each test in an isolated temp directory with a fresh Writers singleton."""
    _reset_writers()
    monkeypatch.chdir(tmp_path)
    yield
    _reset_writers()
    # Destroy closed ParquetWriter objects while libarrow is still healthy.
    # Leaving them for interpreter shutdown can SIGSEGV on Linux CI (exit 139).
    gc.collect()


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
