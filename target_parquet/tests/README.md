# target-parquet test suite

## Files

| File | Scope | What it covers |
|---|---|---|
| `conftest.py` | All tests | Autouse fixture for test isolation (fresh `Writers` singleton, temp working directory). Also holds shared Singer pipeline helpers used by the integration test files. |
| `test_core.py` | Integration | Runs the Singer SDK's built-in standard target tests — the happy-path contract check. |
| `test_integration.py` | Integration | Full Singer pipeline edge cases: fuzzy types, nulls in non-nullable columns, arrays, objects, schema evolution, STATE messages, multiple streams, missing/extra record fields, and multi-batch flushing. |
| `test_integration_types.py` | Integration | Type-specific pipelines (boolean, integer, float, datetime) and config-driven behaviour (`fixed_headers`, `strict_validation`). |

---

## Known Bugs and Bad Behaviours

Issues confirmed by the test suite. None of these raise at the point where the bad input is received — they either surface later or produce silent data corruption.

---

### BUG-1: `anyOf` nullability is silently dropped

`build_pyarrow_field` replaces the full property dict with `anyOf[0]` and then applies its standard nullability logic. Any subsequent variants (including `{"type": "null"}`) are discarded. A field like `{"anyOf": [{"type": "number"}, {"type": "null"}]}` is built as a non-nullable `float64`. When a `None` value is written into it, PyArrow silently coerces it to `0.0` instead of storing null — no exception is raised at write or read time.

Minimal reproducer:
```
SCHEMA  products  price: {"anyOf": [{"type": "number"}, {"type": "null"}]}
RECORD  products  {"id": "1", "price": 9.99}
RECORD  products  {"id": "2", "price": null}   ← stored as 0.0, not null
```

Related tests:
- `TestAnyOfSchemaIntegration::test_anyof_null_variant_loses_nullability`

---

### BUG-2: Type order in fuzzy arrays causes a schema/parse mismatch

`build_pyarrow_field` and `parse_record_value` resolve fuzzy type arrays differently. `build_pyarrow_field` applies a priority chain (boolean > string > first element), so `["number", "string"]` maps to a `string` column. `parse_record_value` uses the first element after stripping `"null"`, so `["number", "string"]` produces a `float`. A float written into a string column is rejected by PyArrow with `ArrowTypeError`. The `["string", "number"]` order works; `["number", "string"]` does not.

Minimal reproducer:
```
SCHEMA  readings  value: {"type": ["number", "string"]}
RECORD  readings  {"id": "1", "value": 42.5}
↑ build_pyarrow_field → string column; parse_record_value → float → ArrowTypeError
```

Related tests:
- `TestFuzzyTypes::test_number_string_order_mismatch`

---

### BUG-3: Schema evolution raises at batch flush, not at schema change time

When a Singer stream sends a second `SCHEMA` message with a different column set, the SDK flushes the pending batch before switching sinks. The first batch is written successfully. The second batch fails because `Writers.start_writer` is idempotent — it reuses the already-open `ParquetWriter` (keyed by stream name) created with the original schema. PyArrow raises `ValueError: Table schema does not match schema used to create file` when the second batch is flushed. The error is deferred and non-obvious.

Minimal reproducer:
```
SCHEMA  contacts  {id, name}
RECORD  contacts  {"id": "1", "name": "Alice"}
SCHEMA  contacts  {id, name, email}            ← column added; writer already open with old schema
RECORD  contacts  {"id": "2", "name": "Bob", "email": "bob@test.com"}  ← raises ValueError
```

Related tests:
- `TestMultipleSchemaMessages::test_schema_with_added_column`
- `TestMultipleSchemaMessages::test_schema_with_removed_column`
