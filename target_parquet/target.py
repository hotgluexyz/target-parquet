"""Parquet target class."""

from typing import Callable

from singer_sdk import typing as th
from singer_sdk.target_base import Target

from target_parquet.checkpoints import append_checkpoint
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
    ).to_dict()
    default_sink_class = ParquetSink

    def _process_state_message(self, message_dict: dict) -> None:
        # Mid-sync: full batches are flushed via drain_one (no state emit).
        # Checkpoint here when the tap's STATE arrives so chunk files match state.
        super()._process_state_message(message_dict)
        append_checkpoint(self._latest_state, Writers()._last_files)

    def _write_state_message(self, state: dict) -> None:
        # After drain_all (end-of-pipe / age drain): flushes a final partial batch
        # and clean_up may rename/combine files. Checkpoint so last_files is the
        # final path — no further STATE message arrives after that.
        super()._write_state_message(state)
        append_checkpoint(state, Writers()._last_files)

    def _process_endofpipe(self) -> None:
        super()._process_endofpipe()
        writers = Writers()
        writers.close_all()


if __name__ == "__main__":
    TargetParquet.cli()
