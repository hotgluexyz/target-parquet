import datetime
import json
import logging
import os
import pathlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("target-parquet")


def get_date_string() -> str:
    return datetime.datetime.now().isoformat()[0:19].replace("-", "").replace(":", "")


def log_milestone(message: str, logger=None) -> None:
    """Log an INFO milestone and flush so the executor sees it promptly."""
    log = logger or LOGGER
    log.info(message)
    for handler in (*log.handlers, *logging.root.handlers):
        try:
            handler.flush()
        except Exception:
            pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Writers(metaclass=SingletonMeta):
    _writers: dict = {}

    def start_writer(self, stream_name: str, schema: pa.Schema):
        if self.exist_writer(stream_name):
            return

        path = f"{stream_name}-{get_date_string()}.parquet"
        self._writers[stream_name] = pq.ParquetWriter(
            path,
            schema,
            compression="ZSTD",
            compression_level=3,
        )
        log_milestone(f"Opened parquet writer for stream '{stream_name}': {path}")

    def close_all(self):
        writers = list(self._writers.values())
        self._writers.clear()
        for value in writers:
            if value.is_open:
                value.close()

    def exist_writer(self, stream_name: str):
        return self._writers.get(stream_name) is not None

    def write(self, stream_name: str, row):
        writer = self._writers.get(stream_name)

        if not writer:
            return

        writer.write(row)

    def update_job_metrics(self, stream_name: str):
        job_metrics_path = "job_metrics.json"

        if not os.path.isfile(job_metrics_path):
            pathlib.Path(job_metrics_path).touch()

        with open(job_metrics_path, "r+") as f:
            content = dict()

            try:
                content = json.loads(f.read())
            except:
                pass

            if not content.get("recordCount"):
                content["recordCount"] = dict()

            content["recordCount"][stream_name] = (
                content["recordCount"].get(stream_name, 0) + 1
            )

            f.seek(0)
            f.write(json.dumps(content))
