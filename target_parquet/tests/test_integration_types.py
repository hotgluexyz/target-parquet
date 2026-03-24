"""Integration tests: type coverage and config-driven behaviour.

Covers:
1. Boolean columns — True/False values, False is not treated as None
2. Integer columns — int values, 0 is not treated as None
3. Number (float) columns — float values, 0.0 is not treated as None
4. Datetime columns — valid parsing, invalid value falls back to None
5. anyOf schema fields — full pipeline round-trip
6. fixed_headers config — selected columns only in output
7. strict_validation config — non-strict silently passes, strict raises
"""

import pyarrow as pa
import pytest

from target_parquet.tests.conftest import (
    read_parquet_for_stream,
    record_message,
    run_target,
    schema_message,
)


class TestBooleanColumns:
    def test_true_and_false_written_correctly(self, tmp_path):
        messages = [
            schema_message("flags", {
                "id": {"type": ["string", "null"]},
                "active": {"type": ["boolean", "null"]},
            }, ["id"]),
            record_message("flags", {"id": "1", "active": True}),
            record_message("flags", {"id": "2", "active": False}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "flags")
        assert table.column("active").to_pylist() == [True, False]
        assert table.schema.field("active").type == pa.bool_()
        assert table.schema.field("active").nullable

    def test_false_is_not_treated_as_none(self, tmp_path):
        """False is falsy but must not be coerced to None by parse_record_value."""
        messages = [
            schema_message("flags", {
                "id": {"type": ["string", "null"]},
                "enabled": {"type": ["boolean", "null"]},
            }, ["id"]),
            record_message("flags", {"id": "1", "enabled": False}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "flags")
        assert table.column("enabled")[0].as_py() is False
        assert table.schema.field("enabled").type == pa.bool_()

    def test_null_boolean_column(self, tmp_path):
        messages = [
            schema_message("flags", {
                "id": {"type": ["string", "null"]},
                "active": {"type": ["boolean", "null"]},
            }, ["id"]),
            record_message("flags", {"id": "1", "active": None}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "flags")
        assert table.column("active")[0].as_py() is None
        assert table.schema.field("active").type == pa.bool_()
        assert table.schema.field("active").nullable


class TestIntegerColumns:
    def test_integer_values_written_correctly(self, tmp_path):
        messages = [
            schema_message("counts", {
                "id": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]},
            }, ["id"]),
            record_message("counts", {"id": "1", "count": 42}),
            record_message("counts", {"id": "2", "count": -7}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "counts")
        assert table.column("count").to_pylist() == [42, -7]
        assert table.schema.field("count").type == pa.int64()
        assert table.schema.field("count").nullable

    def test_zero_integer_is_not_treated_as_none(self, tmp_path):
        """0 is falsy but must survive the full pipeline as a valid integer."""
        messages = [
            schema_message("counts", {
                "id": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]},
            }, ["id"]),
            record_message("counts", {"id": "1", "count": 0}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "counts")
        assert table.column("count")[0].as_py() == 0
        assert table.schema.field("count").type == pa.int64()

    def test_string_coerced_to_integer(self, tmp_path):
        messages = [
            schema_message("counts", {
                "id": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]},
            }, ["id"]),
            record_message("counts", {"id": "1", "count": "99"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "counts")
        assert table.column("count")[0].as_py() == 99
        assert table.schema.field("count").type == pa.int64()


class TestNumberColumns:
    def test_float_values_written_correctly(self, tmp_path):
        messages = [
            schema_message("prices", {
                "id": {"type": ["string", "null"]},
                "price": {"type": ["number", "null"]},
            }, ["id"]),
            record_message("prices", {"id": "1", "price": 19.99}),
            record_message("prices", {"id": "2", "price": -0.5}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "prices")
        assert table.column("price").to_pylist() == [19.99, -0.5]
        assert table.schema.field("price").type == pa.float64()
        assert table.schema.field("price").nullable

    def test_zero_float_is_not_treated_as_none(self, tmp_path):
        """0.0 is falsy but must survive the full pipeline as a valid float."""
        messages = [
            schema_message("prices", {
                "id": {"type": ["string", "null"]},
                "price": {"type": ["number", "null"]},
            }, ["id"]),
            record_message("prices", {"id": "1", "price": 0.0}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "prices")
        assert table.column("price")[0].as_py() == 0.0
        assert table.schema.field("price").type == pa.float64()


class TestDatetimeColumns:
    def test_valid_datetime_string_parsed_to_timestamp(self, tmp_path):
        messages = [
            schema_message("events", {
                "id": {"type": ["string", "null"]},
                "created_at": {"type": ["string", "null"], "format": "date-time"},
            }, ["id"]),
            record_message("events", {"id": "1", "created_at": "2024-06-15T12:00:00Z"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "events")
        ts_field = table.schema.field("created_at")
        assert ts_field.type == pa.timestamp("ms")
        assert ts_field.nullable
        ts = table.column("created_at")[0].as_py()
        assert ts is not None
        assert ts.year == 2024 and ts.month == 6 and ts.day == 15

    def test_invalid_datetime_becomes_null(self, tmp_path):
        """datetime_error_treatment = NULL means malformed timestamps are coerced to None."""
        messages = [
            schema_message("events", {
                "id": {"type": ["string", "null"]},
                "created_at": {"type": ["string", "null"], "format": "date-time"},
            }, ["id"]),
            record_message("events", {"id": "1", "created_at": "not-a-date"}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "events")
        assert table.column("created_at")[0].as_py() is None

    def test_null_datetime_column(self, tmp_path):
        messages = [
            schema_message("events", {
                "id": {"type": ["string", "null"]},
                "created_at": {"type": ["string", "null"], "format": "date-time"},
            }, ["id"]),
            record_message("events", {"id": "1", "created_at": None}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "events")
        assert table.column("created_at")[0].as_py() is None


class TestAnyOfSchemaIntegration:
    def test_anyof_non_null_value_written_correctly(self, tmp_path):
        """anyOf schema fields (common in Singer taps) resolve to the first variant."""
        messages = [
            schema_message("products", {
                "id": {"type": ["string", "null"]},
                "price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            }, ["id"]),
            record_message("products", {"id": "1", "price": 9.99}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "products")
        assert table.num_rows == 1
        assert table.column("price")[0].as_py() == 9.99
        assert table.schema.field("price").type == pa.float64()

    def test_anyof_null_variant_loses_nullability(self, tmp_path):
        """BUG: build_pyarrow_field only inspects anyOf[0] for the type and discards
        subsequent null variants, building the field as non-nullable float64.
        PyArrow then coerces None → 0.0 silently instead of writing null."""
        messages = [
            schema_message("products", {
                "id": {"type": ["string", "null"]},
                "price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            }, ["id"]),
            record_message("products", {"id": "1", "price": None}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "products")
        assert not table.schema.field("price").nullable
        assert table.column("price")[0].as_py() == 0.0


class TestFixedHeadersConfig:
    def test_only_listed_columns_appear_in_output(self, tmp_path):
        config = {"fixed_headers": {"contacts": ["id", "name"]}}
        messages = [
            schema_message("contacts", {
                "id": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
                "email": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("contacts", {"id": "1", "name": "Alice", "email": "alice@test.com"}),
        ]
        run_target(messages, config=config)
        table = read_parquet_for_stream(tmp_path, "contacts")
        assert set(table.column_names) == {"id", "name"}
        assert "email" not in table.column_names
        assert table.column("id")[0].as_py() == "1"
        assert table.column("name")[0].as_py() == "Alice"

    def test_stream_not_in_fixed_headers_is_unaffected(self, tmp_path):
        """fixed_headers only applies to streams explicitly listed in the config."""
        config = {"fixed_headers": {"other_stream": ["id"]}}
        messages = [
            schema_message("contacts", {
                "id": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            }, ["id"]),
            record_message("contacts", {"id": "1", "name": "Alice"}),
        ]
        run_target(messages, config=config)
        table = read_parquet_for_stream(tmp_path, "contacts")
        assert set(table.column_names) == {"id", "name"}
        assert table.column("id")[0].as_py() == "1"
        assert table.column("name")[0].as_py() == "Alice"


class TestEmptyStringToNullConfig:
    """empty_string_to_null config flag controls empty string handling."""

    _PROPS = {
        "id": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
        "count": {"type": ["integer", "null"]},
    }

    def test_empty_string_becomes_null_when_flag_is_true(self, tmp_path):
        """With empty_string_to_null=True (default), '' is written as null for all types."""
        messages = [
            schema_message("users", self._PROPS, ["id"]),
            record_message("users", {"id": "1", "name": "", "count": None}),
        ]
        run_target(messages, config={"empty_string_to_null": True})
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.column("name")[0].as_py() is None

    def test_empty_string_preserved_when_flag_is_false(self, tmp_path):
        """With empty_string_to_null=False, '' is preserved as '' for string columns."""
        messages = [
            schema_message("users", self._PROPS, ["id"]),
            record_message("users", {"id": "1", "name": "", "count": None}),
        ]
        run_target(messages, config={"empty_string_to_null": False})
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.column("name")[0].as_py() == ""

    def test_empty_string_on_non_string_still_null_when_flag_is_false(self, tmp_path):
        """With empty_string_to_null=False, '' on an integer column is still null."""
        messages = [
            schema_message("users", self._PROPS, ["id"]),
            record_message("users", {"id": "1", "name": "Alice", "count": ""}),
        ]
        run_target(messages, config={"empty_string_to_null": False})
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.column("count")[0].as_py() is None

    def test_empty_string_to_null_defaults_to_true(self, tmp_path):
        """Without the flag, empty strings are converted to null (backward-compatible default)."""
        messages = [
            schema_message("users", self._PROPS, ["id"]),
            record_message("users", {"id": "1", "name": "", "count": None}),
        ]
        run_target(messages)
        table = read_parquet_for_stream(tmp_path, "users")
        assert table.column("name")[0].as_py() is None


class TestStrictValidationConfig:
    """_validate_and_parse strict_validation flag controls exception propagation."""

    _PROPS = {
        "id": {"type": "string"},
        "status": {"type": "string", "enum": ["active", "inactive"]},
    }

    def test_non_strict_passes_invalid_record(self, tmp_path):
        """Without strict_validation the schema violation is swallowed and the record is written."""
        messages = [
            schema_message("events", self._PROPS, ["id"]),
            record_message("events", {"id": "1", "status": "invalid-value"}),
        ]
        run_target(messages, config={"strict_validation": False})
        table = read_parquet_for_stream(tmp_path, "events")
        assert table.num_rows == 1
        assert table.column("id")[0].as_py() == "1"
        assert table.column("status")[0].as_py() == "invalid-value"

    def test_strict_raises_on_invalid_record(self, tmp_path):
        """With strict_validation=True the schema violation propagates as an exception."""
        messages = [
            schema_message("events", self._PROPS, ["id"]),
            record_message("events", {"id": "1", "status": "invalid-value"}),
        ]
        with pytest.raises(Exception):
            run_target(messages, config={"strict_validation": True})
