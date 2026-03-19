"""Unit tests for sinks module pure functions and ParquetSink methods."""

import datetime
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from target_parquet.sinks import (
    ParquetSink,
    build_pyarrow_field,
    get_pyarrow_type,
    parse_record_value,
    remove_null_string,
)
from target_parquet.target import TargetParquet
from target_parquet.writers import Writers


def make_sink(schema_props, key_props=None, config=None):
    target = TargetParquet(config=config or {})
    schema = {"type": "object", "properties": schema_props}
    return ParquetSink(target, "test_stream", schema, key_props or [])


class TestRemoveNullString:
    def test_removes_null_from_list(self):
        assert remove_null_string(["string", "null"]) == ["string"]

    def test_preserves_non_null_types(self):
        assert remove_null_string(["string", "number"]) == ["string", "number"]

    def test_only_null(self):
        assert remove_null_string(["null"]) == []

    def test_non_list_passthrough(self):
        assert remove_null_string("string") == "string"

    def test_empty_list(self):
        assert remove_null_string([]) == []

    def test_multiple_nulls(self):
        assert remove_null_string(["null", "string", "null"]) == ["string"]


class TestGetPyarrowType:
    def test_null(self):
        assert get_pyarrow_type("null") == pa.null()

    def test_number(self):
        assert get_pyarrow_type("number") == pa.float64()

    def test_integer(self):
        assert get_pyarrow_type("integer") == pa.int64()

    def test_boolean(self):
        assert get_pyarrow_type("boolean") == pa.bool_()

    def test_string(self):
        assert get_pyarrow_type("string") == pa.string()

    def test_datetime_format(self):
        assert get_pyarrow_type("string", format="date-time") == pa.timestamp("ms")

    def test_array_defaults_to_string(self):
        assert get_pyarrow_type("array") == pa.string()

    def test_object_defaults_to_string(self):
        assert get_pyarrow_type("object") == pa.string()


class TestBuildPyarrowField:
    def test_simple_nullable_string(self):
        field = build_pyarrow_field("name", {"type": ["string", "null"]})
        assert field.name == "name"
        assert field.type == pa.string()
        assert field.nullable

    def test_non_nullable_integer(self):
        field = build_pyarrow_field("count", {"type": "integer"})
        assert field.type == pa.int64()
        assert not field.nullable

    def test_non_nullable_string(self):
        field = build_pyarrow_field("name", {"type": "string"})
        assert field.type == pa.string()
        assert not field.nullable

    def test_anyof_uses_first_variant(self):
        field = build_pyarrow_field("val", {"anyOf": [{"type": "number"}, {"type": "null"}]})
        assert field.type == pa.float64()

    def test_fuzzy_string_number(self):
        """Fuzzy types: string takes priority over number."""
        field = build_pyarrow_field("mixed", {"type": ["string", "number"]})
        assert field.type == pa.string()

    def test_fuzzy_number_string(self):
        """Fuzzy types: string takes priority regardless of order."""
        field = build_pyarrow_field("mixed", {"type": ["number", "string"]})
        assert field.type == pa.string()

    def test_fuzzy_boolean_string(self):
        """Boolean takes priority over string in fuzzy types."""
        field = build_pyarrow_field("flag", {"type": ["boolean", "string"]})
        assert field.type == pa.bool_()

    def test_datetime_is_always_nullable(self):
        """date-time format forces nullable regardless of type array."""
        field = build_pyarrow_field("ts", {"type": "string", "format": "date-time"})
        assert field.type == pa.timestamp("ms")
        assert field.nullable

    def test_datetime_nullable_with_null_type(self):
        field = build_pyarrow_field("ts", {"type": ["string", "null"], "format": "date-time"})
        assert field.type == pa.timestamp("ms")
        assert field.nullable

    def test_array_type_nullable(self):
        field = build_pyarrow_field("tags", {"type": ["array", "null"]})
        assert field.nullable

    def test_object_type_nullable(self):
        field = build_pyarrow_field("meta", {"type": ["object", "null"]})
        assert field.nullable

    def test_no_type_defaults_to_string_null(self):
        """Missing type defaults to ["string", "null"]."""
        field = build_pyarrow_field("unknown", {})
        assert field.type == pa.string()
        assert field.nullable


class TestParseRecordValue:
    def test_none_returns_none(self):
        assert parse_record_value(None, "f", {"type": "string"}) is None

    def test_number_coercion_from_int(self):
        result = parse_record_value(42, "price", {"type": "number"})
        assert result == 42.0
        assert isinstance(result, float)

    def test_number_coercion_from_string(self):
        assert parse_record_value("3.14", "price", {"type": "number"}) == 3.14

    def test_integer_coercion_from_string(self):
        result = parse_record_value("42", "count", {"type": "integer"})
        assert result == 42
        assert isinstance(result, int)

    def test_integer_values_with_string_schema(self):
        result = parse_record_value(42, "id", {"type": "string"})
        assert result == "42"
        assert isinstance(result, str)

    def test_float_value_with_string_schema(self):
        result = parse_record_value(3.14, "val", {"type": "string"})
        assert result == "3.14"

    def test_boolean_value_with_string_schema(self):
        result = parse_record_value(True, "flag", {"type": "string"})
        assert result == "True"

    def test_string_passthrough(self):
        assert parse_record_value("hello", "name", {"type": "string"}) == "hello"

    def test_empty_string_non_string_returns_none(self):
        assert parse_record_value("", "count", {"type": "integer"}) is None
        assert parse_record_value("", "price", {"type": "number"}) is None

    def test_empty_string_string_type_preserved(self):
        assert parse_record_value("", "name", {"type": "string"}) == ""

    def test_datetime_string_parsed(self):
        result = parse_record_value(
            "2024-01-15T10:30:00Z",
            "ts",
            {"type": "string", "format": "date-time"},
        )
        assert isinstance(result, datetime.datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_datetime_object_passthrough(self):
        dt = datetime.datetime(2024, 1, 15, 10, 30)
        result = parse_record_value(dt, "ts", {"type": "string", "format": "date-time"})
        assert result is dt

    def test_list_of_primitives_serialized(self):
        result = parse_record_value([1, 2, 3], "tags", {"type": "array"})
        assert json.loads(result) == [1, 2, 3]

    def test_list_of_strings_serialized(self):
        result = parse_record_value(["a", "b"], "tags", {"type": "array"})
        assert json.loads(result) == ["a", "b"]

    def test_list_of_objects_serialized(self):
        data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        result = parse_record_value(data, "items", {"type": "array"})
        assert json.loads(result) == data

    def test_dict_serialized_to_json(self):
        obj = {"key": "value", "count": 42}
        result = parse_record_value(obj, "meta", {"type": "object"})
        assert json.loads(result) == obj

    def test_stringified_object_passthrough(self):
        stringified = '{"key": "value"}'
        result = parse_record_value(stringified, "data", {"type": "string"})
        assert result == stringified

    def test_anyof_uses_first_variant(self):
        result = parse_record_value(42, "val", {"anyOf": [{"type": "number"}]})
        assert result == 42.0

    def test_no_type_defaults_to_string(self):
        result = parse_record_value(42, "f", {})
        assert result == "42"

    def test_fuzzy_type_uses_first_non_null(self):
        """Fuzzy types: parse_record_value uses the first type after removing null."""
        result = parse_record_value("42", "val", {"type": ["number", "null"]})
        assert result == 42.0

    def test_nullable_number_with_none(self):
        result = parse_record_value(None, "val", {"type": ["number", "null"]})
        assert result is None


class TestParquetSinkStartBatch:
    def test_builds_schema_from_properties(self):
        sink = make_sink({"id": {"type": ["string", "null"]}, "count": {"type": ["integer", "null"]}})
        context = {}
        sink.start_batch(context)
        assert context["schema"].get_field_index("id") >= 0
        assert context["schema"].get_field_index("count") >= 0
        assert context["schema"].field("id").type == pa.string()
        assert context["schema"].field("count").type == pa.int64()

    def test_fixed_headers_filters_columns(self):
        sink = make_sink(
            {"id": {"type": ["string", "null"]}, "name": {"type": ["string", "null"]}},
            config={"fixed_headers": {"test_stream": ["id"]}},
        )
        context = {}
        sink.start_batch(context)
        assert context["schema"].get_field_index("id") >= 0
        assert context["schema"].get_field_index("name") == -1

    def test_initializes_empty_records_list(self):
        sink = make_sink({"id": {"type": ["string", "null"]}})
        context = {}
        sink.start_batch(context)
        assert context["records"] == []


class TestParquetSinkProcessRecord:
    def test_coerces_types_via_parse_record_value(self):
        sink = make_sink({"id": {"type": ["string", "null"]}, "count": {"type": ["integer", "null"]}})
        context = {}
        sink.start_batch(context)
        sink.process_record({"id": "1", "count": "42"}, context)
        assert context["records"] == [{"id": "1", "count": 42}]

    def test_missing_field_becomes_none(self):
        sink = make_sink({"id": {"type": ["string", "null"]}, "name": {"type": ["string", "null"]}})
        context = {}
        sink.start_batch(context)
        sink.process_record({"id": "1"}, context)
        assert context["records"][0]["name"] is None

    def test_updates_job_metrics(self, tmp_path):
        sink = make_sink({"id": {"type": ["string", "null"]}})
        context = {}
        sink.start_batch(context)
        sink.process_record({"id": "1"}, context)
        sink.process_record({"id": "2"}, context)
        data = json.loads((tmp_path / "job_metrics.json").read_text())
        assert data["recordCount"]["test_stream"] == 2


class TestParquetSinkProcessBatch:
    def test_writes_table_to_parquet(self, tmp_path):
        sink = make_sink({"id": {"type": ["string", "null"]}})
        context = {}
        sink.start_batch(context)
        sink.process_record({"id": "1"}, context)
        sink.process_record({"id": "2"}, context)
        sink.process_batch(context)
        Writers().close_all()
        table = pq.read_table(list(tmp_path.glob("test_stream-*.parquet"))[0])
        assert table.num_rows == 2
        assert table.column("id").to_pylist() == ["1", "2"]


class TestParquetSinkValidation:
    """_validate_and_parse wraps the SDK validator with strict/non-strict modes."""

    _SCHEMA = {"id": {"type": "string"}, "status": {"type": "string", "enum": ["active", "inactive"]}}

    def test_non_strict_swallows_validation_error(self):
        """In non-strict mode (default), schema violations return the raw record."""
        sink = make_sink(self._SCHEMA)
        record = {"id": "1", "status": "invalid-value"}
        result = sink._validate_and_parse(record)
        assert result is not None

    def test_strict_reraises_validation_error(self):
        """In strict mode, schema violations propagate as exceptions."""
        sink = make_sink(self._SCHEMA, config={"strict_validation": True})
        with pytest.raises(Exception):
            sink._validate_and_parse({"id": "1", "status": "invalid-value"})
