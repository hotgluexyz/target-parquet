# target-parquet test suite

## Files

| File | Scope | What it covers |
|---|---|---|
| `conftest.py` | All tests | Autouse fixture for test isolation (fresh `Writers` singleton, temp working directory). Also holds shared Singer pipeline helpers used by the integration test files. |
| `test_core.py` | Integration | Runs the Singer SDK's built-in standard target tests — the happy-path contract check. |
| `test_sinks.py` | Unit | Pure functions in `sinks.py` (`remove_null_string`, `get_pyarrow_type`, `build_pyarrow_field`, `parse_record_value`) and `ParquetSink` methods (`start_batch`, `process_record`, `process_batch`, `_validate_and_parse`). |
| `test_writers.py` | Unit | `Writers` singleton class: file creation, idempotency, write accumulation, `close_all`, and `job_metrics.json` accounting. |
| `test_integration.py` | Integration | Full Singer pipeline edge cases: fuzzy types, nulls in non-nullable columns, arrays, objects, schema evolution, STATE messages, multiple streams, missing/extra record fields, and multi-batch flushing. |
| `test_integration_types.py` | Integration | Type-specific pipelines (boolean, integer, float, datetime) and config-driven behaviour (`fixed_headers`, `strict_validation`). |

---

## Known Bugs and Bad Behaviours

Issues confirmed by the test suite. None of these raise at the point where the bad input is received — they either surface later or produce silent data corruption.

---

### BUG-1: `anyOf` nullability is silently dropped

`build_pyarrow_field` replaces the full property dict with `anyOf[0]` and then applies its standard nullability logic. Any subsequent variants (including `{"type": "null"}`) are discarded. A field like `{"anyOf": [{"type": "number"}, {"type": "null"}]}` is built as a non-nullable `float64`. When a `None` value is written into it, PyArrow silently coerces it to `0.0` instead of storing null — no exception is raised at write or read time.

Related tests:
- `TestAnyOfSchemaIntegration::test_anyof_null_variant_loses_nullability`

---

### BUG-2: Non-nullable column with `None` value produces a file that PyArrow cannot read back

When a field is declared as `{"type": "string"}` (no `"null"` in the type array) and the record supplies `None`, `parse_record_value` correctly returns `None`. PyArrow writes the file without raising, but reading it back via `pq.read_table()` fails with `OSError: Unexpected end of stream`. The file is not fully corrupt — external viewers (e.g. parquet-viewer) can still open it — but it is unreadable by PyArrow.

Related tests:
- `TestNonNullColumnsWithNullValues::test_non_nullable_column_with_null_value`
- `TestParquetSinkProcessRecord::test_missing_field_becomes_none`

---

### BUG-3: Type order in fuzzy arrays causes a schema/parse mismatch

`build_pyarrow_field` and `parse_record_value` resolve fuzzy type arrays differently. `build_pyarrow_field` applies a priority chain (boolean > string > first element), so `["number", "string"]` maps to a `string` column. `parse_record_value` uses the first element after stripping `"null"`, so `["number", "string"]` produces a `float`. A float written into a string column is rejected by PyArrow with `ArrowTypeError`. The `["string", "number"]` order works; `["number", "string"]` does not.

Related tests:
- `TestFuzzyTypes::test_number_string_order_mismatch`
- `TestBuildPyarrowField::test_fuzzy_number_string`
- `TestParseRecordValue::test_fuzzy_type_uses_first_non_null`

---

### BUG-4: Schema evolution raises at batch flush, not at schema change time

When a Singer stream sends a second `SCHEMA` message with a different column set, the SDK flushes the pending batch before switching sinks. The first batch is written successfully. The second batch fails because `Writers.start_writer` is idempotent — it reuses the already-open `ParquetWriter` (keyed by stream name) created with the original schema. PyArrow raises `ValueError: Table schema does not match schema used to create file` when the second batch is flushed. The error is deferred and non-obvious.

Related tests:
- `TestMultipleSchemaMessages::test_schema_with_added_column`
- `TestMultipleSchemaMessages::test_schema_with_removed_column`
