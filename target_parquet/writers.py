import json
import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
from target_parquet import util



class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Writers(metaclass=SingletonMeta):
    _writers: dict = {}
    _schemas: dict = {}
    _batch_count: dict = {}

    def start_schema(self, stream_name: str, schema: pa.Schema):
        if self.exist_schema(stream_name):
            return

        self._schemas[stream_name] = schema

    def start_writer(self, stream_name: str):
        if self.exist_writer(stream_name):
            return

        schema = self._schemas[stream_name]

        if self._batch_count.get(stream_name) is None:
            self._batch_count[stream_name] = -1

        self._batch_count[stream_name] += 1

        self._writers[stream_name] = pq.ParquetWriter(
            f"{stream_name}-{util.get_date_string()}-{self._batch_count[stream_name]}.parquet",
            schema
        )

    def close_one(self, stream_name: str):
        if not self.exist_writer(stream_name):
            return

        writer = self._writers.pop(stream_name, None)
        
        if writer is not None and writer.is_open:
            writer.close()

    def close_all(self):
        for value in self._writers.values():
            value.close()

        self._writers = {}

    def exist_writer(self, stream_name: str):
        return self._writers.get(stream_name) is not None

    def exist_schema(self, stream_name: str):
        return self._schemas.get(stream_name) is not None

    def write(self, stream_name: str, row):
        self.start_writer(stream_name)

        writer = self._writers.get(stream_name)

        if not writer:
            return

        writer.write(row)

        if writer.is_open:
            writer.close()
            self._writers.pop(stream_name)

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
