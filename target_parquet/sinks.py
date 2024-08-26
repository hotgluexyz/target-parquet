"""Parquet target sink class, which handles writing streams."""
import datetime
import json
import os
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq
from dateutil import parser as datetime_parser
from jsonschema import FormatChecker
from singer_sdk.sinks import BatchSink

from target_parquet.validator import ParquetValidator
from target_parquet.writers import Writers
from target_parquet import util
from singer_sdk.helpers._typing import DatetimeErrorTreatmentEnum


def remove_null_string(array: list):
    return list(filter(lambda e: e != "null", array))


def get_pyarrow_type(type_id: str, format=None):
    if type_id == "null":
        return pa.null()

    if type_id == "number":
        return pa.float32()

    if type_id == "integer":
        return pa.int64()

    if type_id == "boolean":
        return pa.bool_()

    if format == "date-time":
        return pa.timestamp("ms")

    return pa.string()


def build_pyarrow_field(key: str, value: dict):
    if "anyOf" in value:
        value = value["anyOf"][0]
    types = value.get("type", ["string", "null"])

    is_nullable = any(i for i in ("null", "array", "object") if i in types) or value.get("format") == "date-time"

    if is_nullable:
        types = remove_null_string(types)

    if isinstance(types, str):
        type_id = types
    elif len(types) == 1:
        type_id = types[0]
    elif "boolean" in types:
        type_id = "boolean"
    elif "string" in types:
        type_id = "string"
    else:
        type_id = types[0]

    return pa.field(
        key, get_pyarrow_type(type_id, value.get("format")), nullable=is_nullable
    )


def parse_record_value(record_value, property: dict):
    if record_value in [None, ""]:
        return None

    if "anyOf" in property:
        property = property["anyOf"][0]

    type_id = remove_null_string(property["type"])[0]

    if type_id == "number":
        return float(record_value)

    if type_id == "integer":
        return int(record_value)

    if type_id == "string" and property.get("format") == "date-time":
        return (
            record_value
            if isinstance(record_value, datetime.datetime)
            else datetime_parser.parse(record_value)
        )

    if type_id == "string":
        return str(record_value)

    if isinstance(record_value, (list, dict)):
        try:
            return json.dumps(record_value, default=str)
        except:
            return str(record_value)

    return record_value


class ParquetSink(BatchSink):
    """Parquet target sink class."""

    @property
    def max_size(self) -> int:
        return self._config.get("BATCH_MAX_SIZE") or 1000

    def __init__(
        self,
        target,
        stream_name,
        schema,
        key_properties,
    ) -> None:
        """Initialize target sink."""
        super().__init__(target, stream_name, schema, key_properties)
        self._validator = ParquetValidator(self.schema, format_checker=None)

    def _validate_and_parse(self, record: Dict) -> Dict:
        try:
            return super()._validate_and_parse(record)
        except Exception as e:
            # NOTE: If the below flag is not on we will silently have typing issues and not report them
            if self._config.get("strict_validation", False):
                self.logger.exception(f"Error validating and parsing record.")
                raise e
            return record

    @property
    def datetime_error_treatment(self) -> DatetimeErrorTreatmentEnum:
        return DatetimeErrorTreatmentEnum.NULL

    def start_batch(self, context: dict) -> None:
        """Start a batch."""
        selected_cols = None
        if self.config and self.config.get("fixed_headers"):
            fixed_headers = self.config['fixed_headers']
            selected_cols = fixed_headers.get(self.stream_name)

        schema = pa.schema(
            [build_pyarrow_field(k, v) for (k, v) in self.schema["properties"].items() if selected_cols is None or k in selected_cols],
            metadata={"key_properties": json.dumps(self.key_properties)},
        )
        context["schema"] = schema
        context["records"] = []

        self.writers = Writers()
        self.writers.start_schema(self.stream_name, schema)

    def process_record(self, record: dict, context: dict) -> None:
        """Process the record."""

        for (key, property) in self.schema["properties"].items():
            record[key] = parse_record_value(record.get(key), property)

        context["records"].append(record)

        self.writers.update_job_metrics(self.stream_name)

    def process_batch(self, context: dict) -> None:

        table = pa.Table.from_pylist(context["records"], schema=context["schema"])
        self.writers.write(self.stream_name, table)

    def __del__(self):
        # This is going to have the following format:
        #
        # { "STREAM_NAME": { "final_file_path": "FILE_PATH", "file_paths": ["FILE PATHS"] } }
        #
        # The "final_file_path" is going to be the file path of the final parquet file (the "combined" one).
        # Ideally its value is going to be the first file path.
        #
        # The "file_paths" field is going to store all the file paths of that stream.
        parquet_files = {}

        # Building "parquet_files"
        for f in sorted(os.listdir(".")):
            if not (os.path.isfile(f) and f.endswith(".parquet")):
                continue

            stream_name = f.split("-")[0]

            # If it's not initialized yet, initialize it and add "f" to "final_file_path"
            if not parquet_files.get(stream_name):
                parquet_files[stream_name] = {
                    "final_file_path": f"{stream_name}-{util.get_date_string()}.parquet",
                    "file_paths": []
                }

            parquet_files[stream_name]["file_paths"].append(f)

        for stream_dict in parquet_files.values():
            file_paths = stream_dict["file_paths"]

            if not file_paths:
                continue

            final_table = pq.read_table(file_paths[0])

            os.remove(file_paths[0])

            file_paths = file_paths[1:]

            for file_path in file_paths:
                table = pq.read_table(file_path)

                final_table = pa.concat_tables([final_table, table])

                os.remove(file_path)

            pq.write_table(final_table, stream_dict["final_file_path"])
