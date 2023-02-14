import datetime

import pyarrow as pa
import pyarrow.parquet as pq


def get_date_string() -> str:
    return datetime.datetime.now().isoformat()[0:19].replace("-", "").replace(":", "")


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

        self._writers[stream_name] = pq.ParquetWriter(
            f"{stream_name}-{get_date_string()}.parquet", schema
        )

    def close_all(self):
        for value in self._writers.values():
            value.close()

        self._writers = {}

    def exist_writer(self, stream_name: str):
        return self._writers.get(stream_name) is not None

    def write(self, stream_name: str, row):
        writer = self._writers.get(stream_name)

        if not writer:
            return

        writer.write(row)
