"""Parquet target class."""

from typing import Callable

from hotglue_singer_sdk import typing as th
from hotglue_singer_sdk.target_base import Target

from target_parquet.sinks import ParquetSink
from target_parquet.writers import Writers


class TargetParquet(Target):
    """Sample target for Parquet."""

    name = "target-parquet"
    config_jsonschema = th.PropertiesList(
        th.Property(
            "filepath", th.StringType, description="The path to the target output file"
        ),
        th.Property(
            "file_naming_scheme",
            th.StringType,
            description="The scheme with which output files will be named",
        ),
        th.Property(
            "BATCH_MAX_SIZE",
            th.IntegerType,
            description="Maximum records per in-memory batch before draining (default 10000)",
        ),
        th.Property(
            "BATCH_MAX_BYTES",
            th.IntegerType,
            description="Maximum estimated row payload bytes per batch before draining (default 64MiB; 0=disabled)",
        ),
    ).to_dict()
    default_sink_class = ParquetSink

    def _process_endofpipe(self) -> None:
        super()._process_endofpipe()
        writers = Writers()
        writers.close_all()


if __name__ == "__main__":
    TargetParquet.cli()
