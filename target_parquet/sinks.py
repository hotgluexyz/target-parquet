"""Parquet target sink class, which handles writing streams."""
import datetime
import json
import logging
from dateutil import parser
from typing import Any, Dict, List, Optional

import pyarrow as pa
from dateutil import parser as datetime_parser
from jsonschema import FormatChecker
from singer_sdk.sinks import BatchSink

from target_parquet.validator import ParquetValidator
from target_parquet.writers import Writers
from singer_sdk.helpers._typing import (
    DatetimeErrorTreatmentEnum,
    get_datelike_property_type,
)

_MAX_TIMESTAMP = "9999-12-31 23:59:59.999999"
_MAX_TIME = "23:59:59.999999"


def remove_null_string(array: list):
    if not isinstance(array, list):
        return array
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


def parse_record_value(record_value, property: dict, logger: logging.Logger):
    if record_value in [None, ""]:
        return None

    if "anyOf" in property:
        property = property["anyOf"][0]

    if "type" in property:
        types = remove_null_string(property["type"])
        type_id = types[0] if isinstance(types, list) else types
    else:
        # raise Exception(f"No type found for property {key}:{property}. Check your schema.")
        type_id = "string"

    if type_id == "number":
        return float(record_value)

    if type_id == "integer":
        return int(record_value)

    if type_id == "string" and property.get("format") == "date-time":
        if isinstance(record_value, datetime.datetime):
            return record_value
        try:
            return datetime_parser.parse(record_value)
        except Exception as e:
            logger.warning(f"Could not parse date-time value: {record_value}. Error: {e}")
            return None
        
    if type_id == "string":
        return str(record_value)

    if isinstance(record_value, (list, dict)):
        try:
            return json.dumps(record_value, default=str)
        except:
            return str(record_value)

    return record_value

def handle_invalid_timestamp_in_record(
        record,
        key_breadcrumb: List[str],
        invalid_value: Any,
        datelike_typename: str,
        ex: Exception,
        treatment: Optional[DatetimeErrorTreatmentEnum],
        logger: logging.Logger,
    ) -> Any:
        """Apply treatment or raise an error for invalid time values, 
        but avoid logging empty string cases."""

        treatment = treatment or DatetimeErrorTreatmentEnum.ERROR
        msg = (
            f"Could not parse value '{invalid_value}' for "
            f"field '{':'.join(key_breadcrumb)}'."
        )

        # Skip logging if the invalid value is an empty string
        if isinstance(invalid_value, str) and invalid_value == "":
            return None if treatment == DatetimeErrorTreatmentEnum.NULL else _MAX_TIMESTAMP

        if treatment == DatetimeErrorTreatmentEnum.MAX:
            logger.warning(f"{msg}. Replacing with MAX value.\n{ex}\n")
            return _MAX_TIMESTAMP if datelike_typename != "time" else _MAX_TIME

        if treatment == DatetimeErrorTreatmentEnum.NULL:
            logger.warning(f"{msg}. {logger.name} Replacing with NULL.\n{ex}\n")
            return None

        raise ValueError(msg)

class ParquetSink(BatchSink):
    """Parquet target sink class."""

    max_size = 10000

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
        self.writers.start_writer(self.stream_name, schema)

    def process_record(self, record: dict, context: dict) -> None:
        """Process the record."""

        for (key, property) in self.schema["properties"].items():
            record[key] = parse_record_value(record.get(key), property, self.logger)

        context["records"].append(record)

        self.writers.update_job_metrics(self.stream_name)

    def process_batch(self, context: dict) -> None:

        table = pa.Table.from_pylist(context["records"], schema=context["schema"])
        self.writers.write(self.stream_name, table)

    def _parse_timestamps_in_record(
        self, record: Dict, schema: Dict, treatment: DatetimeErrorTreatmentEnum
    ) -> None:
        """Parse strings to datetime.datetime values, repairing or erroring on failure.

        Attempts to parse every field that is of type date/datetime/time. If its value
        is out of range, repair logic will be driven by the `treatment` input arg:
        MAX, NULL, or ERROR.

        Args:
            record: Individual record in the stream.
            schema: TODO
            treatment: TODO
        """
        for key in record.keys():
            datelike_type = get_datelike_property_type(schema["properties"][key])
            if datelike_type:
                try:
                    date_val = record[key]
                    if record[key] is not None:
                        date_val = parser.parse(date_val)
                except Exception as ex:
                    date_val = handle_invalid_timestamp_in_record(
                        record,
                        [key],
                        date_val,
                        datelike_type,
                        ex,
                        treatment,
                        self.logger,
                    )
                record[key] = date_val
