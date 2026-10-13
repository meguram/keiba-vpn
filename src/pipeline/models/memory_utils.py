"""学習パイプライン用のメモリ節約ユーティリティ（結果を変えない範囲のもの）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


def rss_mb() -> float:
    """このプロセスの常駐メモリ(MB)。Linux 以外は NaN。"""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except OSError:
        return float("nan")


def frame_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1e6


def downcast_frame(df: pd.DataFrame, *, keep: Iterable[str] = (), max_category_ratio: float = 0.5) -> pd.DataFrame:
    """
    float64→float32、整数の縮小、重複の多い文字列列→category にする。
    keep の列は触らない（ID・結合キーなど）。文字列以外が混ざる object 列（リスト・辞書）も触らない。
    """
    keep = set(keep)
    out = df.copy()
    for col in out.columns:
        if col in keep:
            continue
        s = out[col]
        if pd.api.types.is_float_dtype(s) and s.dtype != np.float32:
            out[col] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == object and pd.api.types.infer_dtype(s, skipna=True) == "string":
            if s.nunique(dropna=True) <= max_category_ratio * max(len(s), 1):
                out[col] = s.astype("category")
    return out


def read_parquet_columns(
    path: str | Path | Sequence[str | Path],
    columns: Sequence[str] | None = None,
    *,
    filter_expr=None,
    float32: bool = True,
) -> pd.DataFrame:
    """
    polars の遅延読み込みで、必要な列・行だけを読んで pandas にする。
    全列（400 列級の母表）を読み込まずに済ませるための入口。
    """
    import polars as pl

    lf = pl.scan_parquet([str(p) for p in path] if isinstance(path, (list, tuple)) else str(path))
    if filter_expr is not None:
        lf = lf.filter(filter_expr)
    if columns is not None:
        lf = lf.select(list(columns))
    if float32:
        lf = lf.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    return lf.collect().to_pandas()


def iter_batches(n: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def batched_predict(predict_fn, X: pd.DataFrame, batch_size: int = 200_000) -> np.ndarray:
    """行ごとに独立な予測を分割して実行する。出力は一括実行と同じ。"""
    parts = [np.asarray(predict_fn(X.iloc[sl])) for sl in iter_batches(len(X), batch_size)]
    return np.concatenate(parts) if parts else np.empty(0)
