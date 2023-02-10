"""Parquet target sink class, which handles writing streams."""


from dateutil import parser as datetime_parser
from singer_sdk.sinks import BatchSink
import datetime
import json
import pyarrow as pa
import pyarrow.parquet as pq
from target_parquet.writers import Writers


def remove_null_string(array: list):
    return list(filter(lambda e: e != "null", array))


def get_pyarrow_type(type_id: str, format=None):
    if type_id == "null":
        return pa.null()

    if type_id == "number":
        return pa.float32()

    if type_id == "integer":
        return pa.int64()

    if format == "date-time":
        return pa.timestamp("ms")

    return pa.string()


def build_pyarrow_field(key: str, value: dict):
    types = value["type"]

    is_nullable = "null" in types

    if is_nullable:
        types = remove_null_string(types)

    type_id = types[0]

    return pa.field(
        key,
        get_pyarrow_type(type_id, value.get("format")),
        nullable=is_nullable
    )


def parse_record_value(record_value, property: dict):
    if record_value is None:
        return None

    type_id = remove_null_string(property["type"])[0]

    if type_id == "number":
        return float(record_value)

    if type_id == "integer":
        return int(record_value)

    if type_id == "string" and property.get("format") == "date-time":
        return (
            record_value if isinstance(record_value, datetime.datetime)
            else datetime_parser.parse(record_value)
        )

    return json.dumps(record_value)


class ParquetSink(BatchSink):
    """Parquet target sink class."""

    max_size = 10000

    def start_batch(self, context: dict) -> None:
        """Start a batch.
        
        Developers may optionally add additional markers to the `context` dict,
        which is unique to this batch.
        """
        schema = pa.schema(
            [
                build_pyarrow_field(k, v)
                for (k, v) in self.schema["properties"].items()
            ],
            # TODO(davi): add metadata
            # metadata={"key_properties": self.schema}
        )
        context["schema"] = schema

        writers = Writers()
        writers.start_writer(self.stream_name, schema)

    def process_record(self, record: dict, context: dict) -> None:
        """Process the record.

        Developers may optionally read or write additional markers within the
        passed `context` dict from the current batch.
        """
        writers = Writers()

        if not writers.exist_writer(self.stream_name):
            return

        for (key, property) in self.schema["properties"].items():
            record[key] = parse_record_value(record.get(key), property)

        table = pa.Table.from_pylist([record], schema=context["schema"])

        writers.write(self.stream_name, table)

    def process_batch(self, context: dict) -> None:
        """Write out any prepped records and return once fully written."""
        pass
    
