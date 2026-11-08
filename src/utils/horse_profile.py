"""
馬ごとの走法プロファイルベクトルを返すユーティリティ。

tp-25 (ts_score/gc_score) および tp-26 (split_ratio/front_ratio/shift_ratio) の
データを参照し、指定した馬の「現時点の走法プロファイル」を辞書形式で返す。

Usage:
    from src.utils.horse_profile import get_horse_profile, compare_horses

    # 馬1頭のプロファイル
    profile = get_horse_profile("2019105213")
    print(profile["ts_score_avg3"])   # 直近3戦のTS平均

    # 日付指定（レース前日までのデータのみ使用）
    profile = get_horse_profile("2019105213", as_of="2024-10-15")

    # 複数馬を一括取得
    profiles = get_horse_profiles(["2019105213", "2018104321"])

    # キャッシュをクリア（データ更新後）
    clear_cache()
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

def _find_root() -> Path:
    p = Path(__file__).resolve()
    for q in p.parents:
        if (q / "AGENTS.md").exists():
            return q
    raise RuntimeError("リポジトリルートが見つかりません")

REPO_ROOT = _find_root()
WORK = REPO_ROOT / "data/local/modeling/training_pipeline"

# 走法プロファイルの構成軸
PROFILE_COLS = [
    # TS/GC スコア (tp-25)
    "ts_score_last1",  # 直前のTSスコア
    "ts_score_avg3",   # 直近3戦平均
    "ts_score_avg5",   # 直近5戦平均
    "gc_score_last1",
    "gc_score_avg3",
    "gc_score_avg5",
    # 走法スタイル (tp-26)
    "split_ratio_last1",  # 前後半比（大きい=前半速い）
    "split_ratio_avg3",
    "split_ratio_avg5",
    "front_ratio_last1",  # 先行度（大きい=前寄り）
    "front_ratio_avg3",
    "front_ratio_avg5",
    "shift_ratio_avg3",   # ポジション変化（負=後退型）
    "rq_compat_dev",      # レース質との相性偏差値
]

# 要約ベクトル（モデル投入・比較用）
SUMMARY_COLS = [
    "ts_score_avg3",
    "gc_score_avg3",
    "split_ratio_avg3",
    "front_ratio_avg3",
    "shift_ratio_avg3",
    "rq_compat_dev",
]


@lru_cache(maxsize=1)
def _load_master() -> pd.DataFrame:
    """全馬の走法データを一括ロードしてキャッシュ。"""
    ls_path = WORK / "lapscores/lapscores.parquet"
    rq_path = WORK / "race_quality/horse_rq_features.parquet"

    ls_cols = ["race_id", "horse_id",
               "ts_score_avg3", "ts_score_avg5", "ts_score_last1",
               "gc_score_avg3",  "gc_score_avg5",  "gc_score_last1"]
    rq_cols = ["race_id", "horse_id",
               "split_ratio_avg3", "split_ratio_avg5", "split_ratio_last1",
               "front_ratio_avg3", "front_ratio_avg5", "front_ratio_last1",
               "shift_ratio_avg3", "rq_compat_dev"]

    # layer_a から race_date と horse_name を取得
    la = pd.read_parquet(
        WORK / "layer_a.parquet",
        columns=["race_id", "horse_id", "horse_name", "date"],
    )

    merged = la.copy()
    if ls_path.exists():
        merged = merged.merge(pd.read_parquet(ls_path, columns=ls_cols),
                              on=["race_id", "horse_id"], how="left")
    if rq_path.exists():
        merged = merged.merge(pd.read_parquet(rq_path, columns=rq_cols),
                              on=["race_id", "horse_id"], how="left")

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["horse_id", "date"]).reset_index(drop=True)
    return merged


def clear_cache() -> None:
    """parquet データを更新したあとにキャッシュをクリアする。"""
    _load_master.cache_clear()


def get_horse_profile(
    horse_id: str,
    as_of: Optional[str] = None,
) -> dict:
    """
    1頭の走法プロファイルを返す。

    Parameters
    ----------
    horse_id : str
    as_of : str, optional
        "YYYY-MM-DD" 形式。この日付より前のデータのみ使用（時系列安全）。
        None なら最新エントリを使用。

    Returns
    -------
    dict
        horse_id, horse_name, last_race_date, n_races,
        ts_score_avg3 … rq_compat_dev の各値（NaN は None）,
        summary_vector: 6次元 ndarray
    """
    master = _load_master()
    h = master[master["horse_id"] == horse_id]

    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        h = h[h["date"] < cutoff]

    if len(h) == 0:
        return {
            "horse_id": horse_id,
            "error": f"データが見つかりません（horse_id={horse_id!r}）",
        }

    last = h.iloc[-1]
    result: dict = {
        "horse_id":       horse_id,
        "horse_name":     last.get("horse_name"),
        "last_race_date": str(last["date"].date()),
        "n_races":        len(h),
    }

    for col in PROFILE_COLS:
        v = last.get(col, float("nan"))
        result[col] = float(v) if pd.notna(v) else None

    # 6次元要約ベクトル（NaN は 0 で埋めて返す）
    vec = np.array(
        [result.get(c) or 0.0 for c in SUMMARY_COLS],
        dtype=np.float32,
    )
    result["summary_vector"] = vec
    result["summary_labels"] = SUMMARY_COLS

    return result


def get_horse_profiles(
    horse_ids: list[str],
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """
    複数の馬のプロファイルを DataFrame で返す。

    summary_vector 列は含まない（数値列のみ）。
    """
    records = []
    for hid in horse_ids:
        p = get_horse_profile(hid, as_of=as_of)
        records.append({k: v for k, v in p.items()
                        if k not in ("summary_vector", "summary_labels")})
    return pd.DataFrame(records)


def find_similar_horses(
    horse_id: str,
    top_n: int = 10,
    as_of: Optional[str] = None,
    min_races: int = 3,
) -> pd.DataFrame:
    """
    走法プロファイルが似ている馬を返す（コサイン類似度）。

    比較対象: as_of 時点で min_races 以上の出走実績がある全馬の最新プロファイル。
    """
    master = _load_master()

    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        master = master[master["date"] < cutoff]

    # 各馬の最新エントリ
    latest = master.sort_values("date").groupby("horse_id").last().reset_index()
    counts = master.groupby("horse_id").size().rename("n_races")
    latest = latest.join(counts, on="horse_id")
    latest = latest[latest["n_races"] >= min_races]

    available = [c for c in SUMMARY_COLS if c in latest.columns]
    mat = latest[available].fillna(0).to_numpy(dtype=np.float32)

    # クエリ馬のベクトル
    ref = get_horse_profile(horse_id, as_of=as_of)
    if "error" in ref:
        raise ValueError(ref["error"])
    q = np.array([ref.get(c) or 0.0 for c in available], dtype=np.float32)

    # コサイン類似度
    q_norm = np.linalg.norm(q)
    mat_norm = np.linalg.norm(mat, axis=1, keepdims=True).clip(1e-9)
    sims = (mat @ q) / (mat_norm.squeeze() * (q_norm + 1e-9))

    latest = latest.copy()
    latest["similarity"] = sims

    # 自身を除外してトップNを返す
    result = (latest[latest["horse_id"] != horse_id]
              .nlargest(top_n, "similarity")
              [["horse_id", "horse_name", "similarity", "n_races"] + available])
    return result.reset_index(drop=True)
