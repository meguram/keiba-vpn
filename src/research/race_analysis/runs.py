"""全レース結果から、分析用の『1 出走 = 1 行』の軽い表を作る（race_result_flat.parquet の必要な列だけ）。

race_result_flat は Cloud Run にも ops/mirror 経由で届く（src/ops/gcs_mirror.py の TREES）ので、新しい配布物は要らない。
ファイルが更新されたら（更新時刻が変わったら）読み直す。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FLAT_DIR = Path(__file__).resolve().parents[3] / "data/page_reference/tables"
# 結果データの提供元（取得元は netkeiba のみ。2026-10-22 のユーザー確認）。画面の出典表示に使う
SOURCE_PROVIDER = "netkeiba.com"
SOURCE_DATASET = "レース結果"
COLUMNS = ["race_id", "date", "venue", "surface", "distance", "finish_position", "jockey_id"]

_lock = threading.Lock()
_cache: dict[str, object] = {"sig": None, "df": None}


def _signature(base: Path) -> tuple:
    return tuple((p.parent.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(base.glob("*/race_result_flat.parquet")))


def load_runs(base: Path | None = None) -> pd.DataFrame:
    """着順が確定している出走だけ（取消・除外は着順が無いので入らない）。jockey_id が無い行は除く。"""
    base = base or FLAT_DIR
    sig = _signature(base)
    with _lock:
        if _cache["sig"] == sig and _cache["df"] is not None:
            return _cache["df"]   # type: ignore[return-value]
        parts = []
        for f in sorted(base.glob("*/race_result_flat.parquet")):
            try:
                parts.append(pd.read_parquet(f, columns=COLUMNS))
            except Exception:
                logger.exception("race_result_flat の読み込みに失敗: %s", f)
        if not parts:
            df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
        else:
            df = pd.concat(parts, ignore_index=True)
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["date"] = df["date"].astype(str).str[:10]
        df = df[df["finish_position"].notna() & (df["finish_position"] > 0) & df["jockey_id"].notna()].copy()
        df["jockey_id"] = df["jockey_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)
        df["finish_position"] = df["finish_position"].astype("int16")
        for c in ("venue", "surface"):
            df[c] = df[c].astype("category")
        _cache["sig"], _cache["df"] = sig, df
        return df
