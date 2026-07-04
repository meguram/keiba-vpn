"""10gen 系インデックス構築用: long 行を巨大 list に溜めず Parquet に追記する。

``build_pedigree_10gen_index`` / ``build_pedigree_10gen_3view_index`` で
数千万行を ``list`` + ``DataFrame`` に二重保持しないためのユーティリティ。
"""
from __future__ import annotations

import gc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_SCHEMA = pa.schema(
    [
        ("horse_id", pa.string()),
        ("ancestor_id", pa.string()),
        ("side", pa.string()),
    ]
)


class LongParquetSink:
    """horse_id / ancestor_id / side の long 表を row group 単位で書き出す。"""

    def __init__(
        self,
        path: Path,
        *,
        max_buffer_rows: int = 800_000,
        compression: str = "zstd",
    ) -> None:
        self.path = path
        self.max_buffer_rows = max(10_000, int(max_buffer_rows))
        self._compression = compression
        self._rows_h: list[str] = []
        self._rows_a: list[str] = []
        self._rows_s: list[str] = []
        self._writer: pq.ParquetWriter | None = None
        self._flushed_row_count = 0

    @property
    def total_rows(self) -> int:
        return self._flushed_row_count + len(self._rows_h)

    def append_triplet(self, horse_id: str, ancestor_id: str, side: str) -> None:
        self._rows_h.append(horse_id)
        self._rows_a.append(ancestor_id)
        self._rows_s.append(side)
        if len(self._rows_h) >= self.max_buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self._rows_h:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.unlink()
            self._writer = pq.ParquetWriter(
                self.path,
                _SCHEMA,
                compression=self._compression,
            )
        n = len(self._rows_h)
        self._writer.write_table(
            pa.table(
                {
                    "horse_id": self._rows_h,
                    "ancestor_id": self._rows_a,
                    "side": self._rows_s,
                },
                schema=_SCHEMA,
            )
        )
        self._flushed_row_count += n
        self._rows_h.clear()
        self._rows_a.clear()
        self._rows_s.clear()
        gc.collect()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {"horse_id": [], "ancestor_id": [], "side": []},
                    schema=_SCHEMA,
                ),
                self.path,
                compression=self._compression,
            )
        gc.collect()
