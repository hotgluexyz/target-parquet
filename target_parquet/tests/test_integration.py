"""Integration tests: full Singer message pipeline through target to Parquet output.

Each class covers a distinct edge case:
1. Records with integer values but string schema
2. Fuzzy types (e.g. ["string", "number"])
3. Non-null columns with null record values
4. Arrays of primitives
5. Arrays of objects
6. Stringified objects
7. Multiple compatible schema messages for the same stream
8. Schema messages without any record messages
9. STATE messages interleaved in the stream
10. Multiple streams in the same file
11. Records with missing or extra fields
12. Batches larger than max_size
"""

import json

import pyarrow as pa
import pytest

from target_parquet.sinks import ParquetSink

from target_parquet.tests.conftest import (
    read_parquet_for_stream,
    record_message,
    run_target,
    schema_message,
    state_message,
)


class TestIntegerValuesWithStringSchema:
    """Edge case 1: records have integer values but schema says string."""

    def test_integers_stored_as_strings(self, tmp_path):
        messages = [
            schema_message("users", {
                "id": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("users", {"id": 100, "name": "Alice"}),
            record_message("users", {"id": 200, "name": "Bob"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.num_rows == 2
        assert table.column("id").to_pylist() == ["100", "200"]
        assert table.schema.field("id").type == pa.string()
        assert table.schema.field("name").type == pa.string()


class TestFuzzyTypes:
    """Edge case 2: fuzzy types like ["string", "number"]."""

    def test_string_number_resolves_to_string(self, tmp_path):
        messages = [
            schema_message("metrics", {
                "id": {"type": ["string", "null"]},
                "value": {"type": ["string", "number"]},
            }, ["id"]),
            record_message("metrics", {"id": "1", "value": "text"}),
            record_message("metrics", {"id": "2", "value": 42}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "metrics")
        assert table.num_rows == 2
        values = table.column("value").to_pylist()
        assert values == ["text", "42"]
        assert table.schema.field("value").type == pa.string()

    def test_number_string_order_mismatch(self, tmp_path):
        """Known inconsistency: build_pyarrow_field resolves ["number", "string"]
        to string (string priority), but parse_record_value uses the first type
        (number) — producing a float value for a string column."""
        messages = [
            schema_message("data", {
                "id": {"type": ["string", "null"]},
                "val": {"type": ["number", "string"]},
            }, ["id"]),
            record_message("data", {"id": "1", "val": 3.14}),
        ]
        with pytest.raises(Exception, match="ArrowTypeError|float"):
            run_target(messages)


class TestNonNullColumnsWithNullValues:
    """Edge case 3: columns typed as non-null but records have null values."""

    def test_nullable_columns_accept_nulls(self, tmp_path):
        messages = [
            schema_message("items", {
                "id": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("items", {"id": "1", "description": None}),
            record_message("items", {"id": "2", "description": None}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "items")
        assert table.num_rows == 2
        assert table.column("description").to_pylist() == [None, None]
        assert table.schema.field("description").type == pa.string()
        assert table.schema.field("description").nullable

    def test_non_nullable_column_with_null_value(self, tmp_path):
        """Null in a non-nullable column causes PyArrow to silently write a corrupt
        parquet file. The target exits with no error, but the file cannot be read back.
        PyArrow gives no warning at write time; the failure only surfaces on read."""
        messages = [
            schema_message("strict", {
                "id": {"type": "string"},
                "required_col": {"type": "string"},
            }, ["id"]),
            record_message("strict", {"id": "1", "required_col": None}),
        ]
        run_target(messages)  # exits 0, no exception
        with pytest.raises(OSError, match="Unexpected end of stream"):
            read_parquet_for_stream(tmp_path, "strict")


class TestArraysOfPrimitives:
    """Edge case 4: arrays of primitives are serialized to JSON strings."""

    def test_int_array(self, tmp_path):
        messages = [
            schema_message("data", {
                "id": {"type": ["string", "null"]},
                "scores": {"type": ["array", "null"]},
            }, ["id"]),
            record_message("data", {"id": "1", "scores": [10, 20, 30]}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "data")
        assert json.loads(table.column("scores")[0].as_py()) == [10, 20, 30]

    def test_string_array(self, tmp_path):
        messages = [
            schema_message("data", {
                "id": {"type": ["string", "null"]},
                "tags": {"type": ["array", "null"]},
            }, ["id"]),
            record_message("data", {"id": "1", "tags": ["red", "blue"]}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "data")
        assert json.loads(table.column("tags")[0].as_py()) == ["red", "blue"]

    def test_null_array(self, tmp_path):
        messages = [
            schema_message("data", {
                "id": {"type": ["string", "null"]},
                "tags": {"type": ["array", "null"]},
            }, ["id"]),
            record_message("data", {"id": "1", "tags": None}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "data")
        assert table.column("tags")[0].as_py() is None


class TestArraysOfObjects:
    """Edge case 5: arrays of objects are serialized to JSON strings."""

    def test_object_array(self, tmp_path):
        items = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        messages = [
            schema_message("orders", {
                "id": {"type": ["string", "null"]},
                "line_items": {"type": ["array", "null"]},
            }, ["id"]),
            record_message("orders", {"id": "1", "line_items": items}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "orders")
        assert json.loads(table.column("line_items")[0].as_py()) == items


class TestStringifiedObjects:
    """Edge case 6: stringified objects stored as strings."""

    def test_stringified_json_preserved(self, tmp_path):
        raw = '{"key": "value", "count": 42}'
        messages = [
            schema_message("events", {
                "id": {"type": ["string", "null"]},
                "payload": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("events", {"id": "1", "payload": raw}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "events")
        assert table.column("payload")[0].as_py() == raw

    def test_object_value_with_object_schema(self, tmp_path):
        """Object value with object type gets serialized to JSON."""
        obj = {"nested": {"deep": True}, "list": [1, 2]}
        messages = [
            schema_message("events", {
                "id": {"type": ["string", "null"]},
                "metadata": {"type": ["object", "null"]},
            }, ["id"]),
            record_message("events", {"id": "1", "metadata": obj}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "events")
        assert json.loads(table.column("metadata")[0].as_py()) == obj


class TestMultipleSchemaMessages:
    """Edge case 7: multiple schema messages that are different but compatible."""

    def test_compatible_schemas_same_columns(self, tmp_path):
        """Same columns across schema messages — should work smoothly."""
        props = {
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
        }
        messages = [
            schema_message("contacts", props, ["id"]),
            record_message("contacts", {"id": "1", "name": "Alice"}),
            schema_message("contacts", props, ["id"]),
            record_message("contacts", {"id": "2", "name": "Bob"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "contacts")
        assert table.num_rows == 2
        assert table.column("id").to_pylist() == ["1", "2"]
        assert table.column("name").to_pylist() == ["Alice", "Bob"]

    def test_schema_with_added_column(self, tmp_path):
        """Second schema adds a column — the SDK flushes the first batch (schema_v1) before
        switching sinks, so the first record lands successfully. The second batch fails because
        Writers reuses the open ParquetWriter (keyed by stream name) and PyArrow rejects writing
        a 3-column table into a file opened with a 2-column schema."""
        schema_v1 = {
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
        }
        schema_v2 = {
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
        }
        messages = [
            schema_message("contacts", schema_v1, ["id"]),
            record_message("contacts", {"id": "1", "name": "Alice"}),
            schema_message("contacts", schema_v2, ["id"]),
            record_message("contacts", {"id": "2", "name": "Bob", "email": "bob@test.com"}),
        ]
        with pytest.raises(ValueError, match="Table schema does not match schema used to create file"):
            run_target(messages)

    def test_schema_with_removed_column(self, tmp_path):
        """Second schema removes a column — same failure mode as the added-column case:
        the first batch (schema_v1, 3 cols) is flushed successfully, then the second batch
        (schema_v2, 2 cols) is rejected by PyArrow because it doesn't match the open file's schema."""
        schema_v1 = {
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
        }
        schema_v2 = {
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
        }
        messages = [
            schema_message("contacts", schema_v1, ["id"]),
            record_message("contacts", {"id": "1", "name": "Alice", "phone": "555-0001"}),
            schema_message("contacts", schema_v2, ["id"]),
            record_message("contacts", {"id": "2", "name": "Bob"}),
        ]
        with pytest.raises(ValueError, match="Table schema does not match schema used to create file"):
            run_target(messages)


class TestSchemaWithoutRecords:
    """Edge case 8: schema messages without any record messages."""

    def test_no_records_no_crash(self, tmp_path):
        messages = [
            schema_message("empty_stream", {
                "id": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            }, ["id"]),
        ]
        run_target(messages)
        parquet_files = list(tmp_path.glob("empty_stream-*.parquet"))
        assert len(parquet_files) == 0

    def test_schema_only_with_other_stream_having_records(self, tmp_path):
        """One stream gets schema only, another gets schema + records."""
        messages = [
            schema_message("no_data", {
                "id": {"type": ["string", "null"]},
            }, ["id"]),
            schema_message("has_data", {
                "id": {"type": ["string", "null"]},
                "value": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("has_data", {"id": "1", "value": "test"}),
        ]
        run_target(messages)
        assert not list(tmp_path.glob("no_data-*.parquet"))
        table = read_parquet_for_stream(tmp_path, "has_data")
        assert table.num_rows == 1
        assert table.column("id")[0].as_py() == "1"
        assert table.column("value")[0].as_py() == "test"


class TestStateMessages:
    """Edge case 9: STATE messages interleaved in the Singer stream."""

    def test_state_before_any_records_does_not_crash(self, tmp_path):
        messages = [
            state_message({"bookmarks": {}}),
            schema_message("users", {"id": {"type": ["string", "null"]}}, ["id"]),
            record_message("users", {"id": "1"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.num_rows == 1
        assert table.column("id")[0].as_py() == "1"

    def test_state_between_records_does_not_affect_output(self, tmp_path):
        messages = [
            schema_message("users", {"id": {"type": ["string", "null"]}}, ["id"]),
            record_message("users", {"id": "1"}),
            state_message({"bookmarks": {"users": {"id": "1"}}}),
            record_message("users", {"id": "2"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.num_rows == 2
        assert table.column("id").to_pylist() == ["1", "2"]


class TestMultipleStreams:
    """Edge case 10: multiple streams in the same Singer file."""

    def test_two_streams_produce_separate_files(self, tmp_path):
        messages = [
            schema_message("users", {"id": {"type": ["string", "null"]}}, ["id"]),
            schema_message("orders", {"id": {"type": ["string", "null"]}, "amount": {"type": ["number", "null"]}}, ["id"]),
            record_message("users", {"id": "u1"}),
            record_message("orders", {"id": "o1", "amount": 99.99}),
        ]
        run_target(messages)
        assert len(list(tmp_path.glob("users-*.parquet"))) == 1
        assert len(list(tmp_path.glob("orders-*.parquet"))) == 1
        users_table = read_parquet_for_stream(tmp_path, "users")
        orders_table = read_parquet_for_stream(tmp_path, "orders")
        assert users_table.column("id")[0].as_py() == "u1"
        assert orders_table.column("id")[0].as_py() == "o1"
        assert orders_table.schema.field("amount").type == pa.float64()
        assert orders_table.column("amount")[0].as_py() == 99.99

    def test_interleaved_records_go_to_correct_streams(self, tmp_path):
        messages = [
            schema_message("users", {"id": {"type": ["string", "null"]}}, ["id"]),
            schema_message("orders", {"id": {"type": ["string", "null"]}}, ["id"]),
            record_message("users", {"id": "u1"}),
            record_message("orders", {"id": "o1"}),
            record_message("users", {"id": "u2"}),
            record_message("orders", {"id": "o2"}),
            record_message("orders", {"id": "o3"}),
        ]
        run_target(messages)
        users_table = read_parquet_for_stream(tmp_path, "users")
        orders_table = read_parquet_for_stream(tmp_path, "orders")
        assert users_table.num_rows == 2
        assert orders_table.num_rows == 3
        assert sorted(users_table.column("id").to_pylist()) == ["u1", "u2"]
        assert sorted(orders_table.column("id").to_pylist()) == ["o1", "o2", "o3"]


class TestRecordFieldMismatch:
    """Edge case 11: records with missing or extra fields relative to the schema."""

    def test_missing_nullable_field_becomes_none(self, tmp_path):
        """A field absent from the record dict is treated as None by record.get()."""
        messages = [
            schema_message("items", {
                "id": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("items", {"id": "1"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "items")
        assert table.num_rows == 1
        assert table.column("description")[0].as_py() is None

    def test_extra_fields_in_record_are_ignored(self, tmp_path):
        """Fields in the record that are not in the schema are dropped by PyArrow
        when constructing the table from the schema-typed column list."""
        messages = [
            schema_message("items", {"id": {"type": ["string", "null"]}}, ["id"]),
            record_message("items", {"id": "1", "undeclared_field": "extra-value"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "items")
        assert table.num_rows == 1
        assert table.column_names == ["id"]


class TestMultiBatch:
    """Edge case 12: Singer streams with more records than max_size flush multiple batches."""

    def test_records_beyond_max_size_all_written(self, tmp_path, monkeypatch):
        """Writers reuses the same open ParquetWriter across batches (start_writer is
        idempotent), so all batches are appended to the same file as separate row groups."""
        monkeypatch.setattr(ParquetSink, "max_size", 3)
        messages = [
            schema_message("data", {"id": {"type": ["string", "null"]}}, ["id"]),
            *[record_message("data", {"id": str(i)}) for i in range(7)],
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "data")
        assert table.num_rows == 7
        assert sorted(table.column("id").to_pylist()) == [str(i) for i in range(7)]
