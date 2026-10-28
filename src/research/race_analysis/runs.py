"""全レース結果から、分析用の『1 出走 = 1 行』の軽い表を作る（race_result_flat.parquet の必要な列だけ）。

race_result_flat は Cloud Run にも ops/mirror 経由で届く（src/ops/gcs_mirror.py の TREES）ので、新しい配布物は要らない。
ファイルが更新されたら（更新時刻が変わったら）読み直す。
種牡馬の対応表（horse_sire_map.parquet）は src/scripts/data/build_horse_sire_map.py が作り、pedigree_race_index/ に置く（同じく ops/mirror で届く）。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 結果データの提供元（取得元は netkeiba のみ。2026-10-22 のユーザー確認）。画面の出典表示に使う
SOURCE_PROVIDER = "netkeiba.com"
SOURCE_DATASET = "レース結果"
COLUMNS = ["race_id", "date", "venue", "surface", "distance", "finish_position", "jockey_id", "bracket_number", "horse_id", "passing_order", "field_size"]

REPO = Path(__file__).resolve().parents[3]
FLAT_DIR = REPO / "data/page_reference/tables"
SIRE_MAP_PATH = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"

_lock = threading.Lock()
_cache: dict[str, object] = {"sig": None, "df": None, "sire_sig": None, "sire_df": None}


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
        df["bracket_number"] = pd.to_numeric(df["bracket_number"], errors="coerce").astype("Int8")
        df["date"] = df["date"].astype(str).str[:10]
        df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce").astype("Int16")
        # 初角の通過順位 = 通過順（"3-3-2-2"）の先頭（公表されている最初のコーナー）。頭数を超える値は不正として欠損にする
        first = pd.to_numeric(df["passing_order"].astype("string").str.extract(r"^(\d+)")[0], errors="coerce")
        first = first.where(first.between(1, df["field_size"].astype("float")))
        df["pos_first"] = first.astype("Int16")
        df = df.drop(columns=["passing_order"])
        df = df[df["finish_position"].notna() & (df["finish_position"] > 0) & df["jockey_id"].notna()].copy()
        df["jockey_id"] = df["jockey_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)
        df["horse_id"] = df["horse_id"].astype(str)
        df["finish_position"] = df["finish_position"].astype("int16")
        for c in ("venue", "surface"):
            df[c] = df[c].astype("category")
        _cache["sig"], _cache["df"] = sig, df
        _cache["sire_sig"], _cache["sire_df"] = None, None     # 結果が変わったら、種牡馬を付けた表も作り直す
        return df


def load_sire_map(path: Path | None = None) -> pd.DataFrame | None:
    """horse_id -> sire_id / sire_name / dam_sire_id / dam_sire_name。無ければ None。"""
    path = path or SIRE_MAP_PATH
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        return None


def sire_map_available(path: Path | None = None) -> bool:
    return (path or SIRE_MAP_PATH).exists()


def load_runs_with_sire() -> pd.DataFrame:
    """load_runs に父（sire_id）を付けた表。対応表の更新時刻が変わったら作り直す。父が分からない出走は sire_id が NaN。"""
    runs = load_runs()
    try:
        st = SIRE_MAP_PATH.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return runs.assign(sire_id=pd.NA)
    with _lock:
        if _cache["sire_sig"] == sig and _cache["sire_df"] is not None:
            return _cache["sire_df"]   # type: ignore[return-value]
    m = load_sire_map()
    if m is None:
        return runs.assign(sire_id=pd.NA)
    m = m[["horse_id", "sire_id"]].dropna().drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    out = runs.merge(m, on="horse_id", how="left")
    out["sire_id"] = out["sire_id"].astype("category")
    with _lock:
        _cache["sire_sig"], _cache["sire_df"] = sig, out
    return out
