"""Parquet target class."""

from typing import Optional, Set

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_record_stream: Optional[str] = None
        self._completed_streams: Set[str] = set()

    def _note_record_stream(self, stream_name: str) -> None:
        """Track sequential stream switches for completed_streams.

        When the active record stream changes from A to B, mark A completed.
        Parent/child interleaving (A→B→A) clears that mark for A when A resumes,
        so the executor's history-based inference remains the source of truth for
        resume; this field is a hint that becomes definitive at endofpipe.
        """
        if not stream_name:
            return
        if self._current_record_stream and self._current_record_stream != stream_name:
            self._completed_streams.add(self._current_record_stream)
        # Stream resumed after another (parent/child): no longer completed.
        self._completed_streams.discard(stream_name)
        self._current_record_stream = stream_name

    def _process_schema_message(self, message_dict: dict) -> None:
        stream_name = message_dict.get("stream")
        if stream_name:
            self._note_record_stream(stream_name)
        super()._process_schema_message(message_dict)

    def _process_record_message(self, message_dict: dict) -> None:
        stream_name = message_dict.get("stream")
        if stream_name:
            self._note_record_stream(stream_name)
        super()._process_record_message(message_dict)

    def _process_state_message(self, message_dict: dict) -> None:
        # Mid-sync: full batches are flushed via drain_one (no state emit).
        # Checkpoint here when the tap's STATE arrives so chunk files match state.
        super()._process_state_message(message_dict)
        append_checkpoint(
            self._latest_state,
            Writers()._last_files,
            completed_streams=sorted(self._completed_streams),
        )

    def _write_state_message(self, state: dict) -> None:
        # After drain_all (end-of-pipe / age drain): flushes a final partial batch
        # and clean_up may rename/combine files. Checkpoint so last_files is the
        # final path — no further STATE message arrives after that.
        super()._write_state_message(state)
        append_checkpoint(
            state,
            Writers()._last_files,
            completed_streams=sorted(self._completed_streams),
        )

    def _process_endofpipe(self) -> None:
        super()._process_endofpipe()
        writers = Writers()
        writers.close_all()
        # Entire sync finished: every stream that wrote files is complete.
        self._completed_streams.update(writers._last_files.keys())
        if self._latest_state:
            append_checkpoint(
                self._latest_state,
                writers._last_files,
                completed_streams=sorted(self._completed_streams),
            )


if __name__ == "__main__":
    TargetParquet.cli()
