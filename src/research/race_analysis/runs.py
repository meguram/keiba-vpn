"""全レース結果から、分析用の『1 出走 = 1 行』の軽い表を作る（race_result_flat.parquet の必要な列だけ）。

race_result_flat は Cloud Run にも ops/mirror 経由で届く（src/ops/gcs_mirror.py の TREES）ので、新しい配布物は要らない。
ファイルが更新されたら（更新時刻が変わったら）読み直す。
種牡馬の対応表（horse_sire_map.parquet）は src/scripts/data/build_horse_sire_map.py が作り、pedigree_race_index/ に置く（同じく ops/mirror で届く）。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# 結果データの提供元（取得元は netkeiba のみ。2026-10-22 のユーザー確認）。画面の出典表示に使う
SOURCE_PROVIDER = "netkeiba.com"
SOURCE_DATASET = "レース結果"
COLUMNS = ["race_id", "date", "venue", "surface", "distance", "finish_position", "jockey_id", "bracket_number", "horse_id", "passing_order", "field_size"]

REPO = Path(__file__).resolve().parents[3]
FLAT_DIR = REPO / "data/page_reference/tables"
SIRE_MAP_PATH = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"

_lock = threading.Lock()
_cache: dict[str, object] = {"sig": None, "df": None, "sire_sig": None, "sire_df": None, "roi_sig": None, "roi_df": None}
RET_COLUMNS = ["race_id", "horse_id", "horse_number", "finish_position", "payoff", "odds"]
ODDS_CUT = 50.0   # 回収率を出す表では、単勝オッズがこれ以上の出走（大穴の外れ値）を集計から除く


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
        _cache["roi_sig"], _cache["roi_df"] = None, None
        return df


def _returns(base: Path) -> pd.DataFrame:
    """race_id・horse_id ごとの単勝オッズと、100 円あたりの単勝・複勝の払戻（/100。ハズレは 0、払戻が読めないレースは NaN）。列が無い年は NaN。"""
    from src.pipeline.features.tp28_workout_stats import runner_returns

    parts = []
    for f in sorted(base.glob("*/race_result_flat.parquet")):
        try:
            have = set(pq.read_schema(f).names)
            r = pd.read_parquet(f, columns=[c for c in RET_COLUMNS if c in have])
        except Exception:
            logger.exception("race_result_flat（払戻）の読み込みに失敗: %s", f)
            continue
        for c in RET_COLUMNS:
            if c not in r.columns:
                r[c] = np.nan if c != "payoff" else None
        r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
        r["horse_number"] = pd.to_numeric(r["horse_number"], errors="coerce")
        r["finish_position"] = pd.to_numeric(r["finish_position"], errors="coerce")
        r["odds"] = pd.to_numeric(r["odds"], errors="coerce")
        payoff = r.dropna(subset=["payoff"]).drop_duplicates("race_id").set_index("race_id")["payoff"].to_dict()
        ok = r["horse_number"].notna()
        rr = pd.DataFrame({"win_ret": np.nan, "place_ret": np.nan}, index=r.index)
        if payoff and ok.any():
            rr.loc[ok] = runner_returns(r.loc[ok], payoff).to_numpy()
        parts.append(pd.concat([r[["race_id", "horse_id", "odds"]], rr], axis=1))
    if not parts:
        return pd.DataFrame({"race_id": [], "horse_id": [], "odds": [], "win_ret": [], "place_ret": []})
    return pd.concat(parts, ignore_index=True).drop_duplicates(["race_id", "horse_id"])


def load_runs_roi(base: Path | None = None) -> pd.DataFrame:
    """load_runs に odds（単勝）・win_ret・place_ret（100 円あたりの払戻/100）を付けた表。結果が更新されたら作り直す。"""
    base = base or FLAT_DIR
    runs = load_runs(base)
    sig = _cache["sig"]
    with _lock:
        if _cache["roi_sig"] == sig and _cache["roi_df"] is not None:
            return _cache["roi_df"]   # type: ignore[return-value]
    out = runs.merge(_returns(base), on=["race_id", "horse_id"], how="left")
    with _lock:
        _cache["roi_sig"], _cache["roi_df"] = sig, out
    return out


def drop_long_shots(df: pd.DataFrame, cut: float = ODDS_CUT) -> pd.DataFrame:
    """単勝オッズが cut 以上の出走を除く（オッズが読めない出走は残す）。"""
    return df[~(df["odds"] >= cut)]


def load_sire_map(path: Path | None = None) -> pd.DataFrame | None:
    """horse_id -> sire_id / sire_name / dam_sire_id / dam_sire_name。無ければ None。"""
    path = path or SIRE_MAP_PATH
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        return None


def sire_map_available(path: Path | None = None) -> bool:
    return (path or SIRE_MAP_PATH).exists()


def load_runs_with_sire(roi: bool = False) -> pd.DataFrame:
    """load_runs（roi=True なら load_runs_roi）に父（sire_id）を付けた表。対応表の更新時刻が変わったら作り直す。父が分からない出走は sire_id が NaN。"""
    runs = load_runs_roi() if roi else load_runs()
    try:
        st = SIRE_MAP_PATH.stat()
        sig = (st.st_mtime_ns, st.st_size, roi)
    except OSError:
        return runs.assign(sire_id=pd.NA, sire_line_id=pd.NA)
    with _lock:
        if _cache["sire_sig"] == sig and _cache["sire_df"] is not None:
            return _cache["sire_df"]   # type: ignore[return-value]
    m = load_sire_map()
    if m is None:
        return runs.assign(sire_id=pd.NA, sire_line_id=pd.NA)
    cols = ["horse_id", "sire_id"] + (["sire_line_id"] if "sire_line_id" in m.columns else [])
    m = m[cols].dropna(subset=["sire_id"]).drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    out = runs.merge(m, on="horse_id", how="left")
    for c in cols[1:]:
        out[c] = out[c].astype("category")
    if "sire_line_id" not in out.columns:
        out["sire_line_id"] = pd.NA
    with _lock:
        _cache["sire_sig"], _cache["sire_df"] = sig, out
    return out
