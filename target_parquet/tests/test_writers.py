"""Unit tests for the Writers singleton class."""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from target_parquet.writers import Writers

_SCHEMA = pa.schema([pa.field("id", pa.string(), nullable=True)])


def _table(*ids):
    return pa.table({"id": list(ids)}, schema=_SCHEMA)


class TestStartWriter:
    def test_creates_parquet_file(self, tmp_path):
        Writers().start_writer("stream", _SCHEMA)
        assert len(list(tmp_path.glob("stream-*.parquet"))) == 1

    def test_idempotent_same_stream(self, tmp_path):
        Writers().start_writer("stream", _SCHEMA)
        first_files = list(tmp_path.glob("stream-*.parquet"))
        Writers().start_writer("stream", _SCHEMA)
        assert list(tmp_path.glob("stream-*.parquet")) == first_files


class TestExistWriter:
    def test_false_before_start(self):
        assert not Writers().exist_writer("stream")

    def test_true_after_start(self):
        Writers().start_writer("stream", _SCHEMA)
        assert Writers().exist_writer("stream")

    def test_false_for_different_stream(self):
        Writers().start_writer("stream_a", _SCHEMA)
        assert not Writers().exist_writer("stream_b")


class TestWrite:
    def test_appends_rows(self, tmp_path):
        w = Writers()
        w.start_writer("stream", _SCHEMA)
        w.write("stream", _table("a", "b"))
        w.close_all()
        table = pq.read_table(list(tmp_path.glob("stream-*.parquet"))[0])
        assert table.column("id").to_pylist() == ["a", "b"]

    def test_multiple_writes_accumulate(self, tmp_path):
        w = Writers()
        w.start_writer("stream", _SCHEMA)
        w.write("stream", _table("a"))
        w.write("stream", _table("b", "c"))
        w.close_all()
        table = pq.read_table(list(tmp_path.glob("stream-*.parquet"))[0])
        assert table.num_rows == 3

    def test_noop_for_unknown_stream(self):
        Writers().write("nonexistent", _table("x"))


class TestCloseAll:
    def test_clears_writers(self):
        w = Writers()
        w.start_writer("stream", _SCHEMA)
        w.close_all()
        assert not w.exist_writer("stream")

    def test_file_readable_after_close(self, tmp_path):
        w = Writers()
        w.start_writer("stream", _SCHEMA)
        w.write("stream", _table("a"))
        w.close_all()
        table = pq.read_table(list(tmp_path.glob("stream-*.parquet"))[0])
        assert table.num_rows == 1


class TestUpdateJobMetrics:
    def test_creates_file_on_first_call(self, tmp_path):
        Writers().update_job_metrics("orders")
        assert (tmp_path / "job_metrics.json").exists()

    def test_increments_count_for_stream(self, tmp_path):
        w = Writers()
        w.update_job_metrics("orders")
        w.update_job_metrics("orders")
        w.update_job_metrics("orders")
        data = json.loads((tmp_path / "job_metrics.json").read_text())
        assert data["recordCount"]["orders"] == 3

    def test_tracks_multiple_streams_independently(self, tmp_path):
        w = Writers()
        w.update_job_metrics("orders")
        w.update_job_metrics("users")
        w.update_job_metrics("users")
        data = json.loads((tmp_path / "job_metrics.json").read_text())
        assert data["recordCount"]["orders"] == 1
        assert data["recordCount"]["users"] == 2

    def test_handles_preexisting_content(self, tmp_path):
        (tmp_path / "job_metrics.json").write_text(json.dumps({"recordCount": {"orders": 5}}))
        Writers().update_job_metrics("orders")
        data = json.loads((tmp_path / "job_metrics.json").read_text())
        assert data["recordCount"]["orders"] == 6

    def test_handles_empty_file(self, tmp_path):
        (tmp_path / "job_metrics.json").write_text("")
        Writers().update_job_metrics("orders")
        data = json.loads((tmp_path / "job_metrics.json").read_text())
        assert data["recordCount"]["orders"] == 1
