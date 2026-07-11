"""
馬の累積獲得賞金推計モジュール

horse_race_history の race_name / finish_position から
JRA 賞金テーブルを参照して各馬の累積獲得賞金（万円）を推計し、
めぐ指数の delta_level_sec フィーチャーとして活用する。

使用箇所:
    src/pipeline/megu_index/compute.py の compute_for_dataframe()
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# JRA 標準賞金テーブル (万円・概算, 2020年以降)
# ─────────────────────────────────────────────────────────────────────────────
# 各グレードの本賞金概算（中央競馬・平地）
# 出典: JRA 競馬施行規程、各年度賞金テーブル参考
# 着順 → 賞金 (万円)
PRIZE_TABLE: dict[str, dict[int, float]] = {
    "新馬":  {1: 500,  2: 200,  3: 125,  4: 75,   5: 50},
    "未勝利": {1: 500,  2: 200,  3: 125,  4: 75,   5: 50},
    "1勝":  {1: 700,  2: 280,  3: 175,  4: 105,  5: 70},
    "2勝":  {1: 1100, 2: 440,  3: 275,  4: 165,  5: 110},
    "3勝":  {1: 1800, 2: 720,  3: 450,  4: 270,  5: 180},
    "L":    {1: 2500, 2: 1000, 3: 625,  4: 375,  5: 250},
    "OP":   {1: 2500, 2: 1000, 3: 625,  4: 375,  5: 250},
    "G3":   {1: 4000, 2: 1600, 3: 1000, 4: 600,  5: 400},
    "G2":   {1: 7500, 2: 3000, 3: 1875, 4: 1125, 5: 750},
    "G1":   {1: 20000, 2: 8000, 3: 5000, 4: 3000, 5: 2000},
}

# グレード正規化マッピング（DB の grade / race_class を PRIZE_TABLE キーへ変換）
_GRADE_NORM: dict[str, str] = {
    "新馬": "新馬",
    "未勝利": "未勝利",
    "1勝": "1勝",
    "2勝": "2勝",
    "3勝": "3勝",
    "L": "L",
    "OP": "OP",
    "G3": "G3",
    "G2": "G2",
    "G1": "G1",
    # 旧表記（2019年以前の開催分）
    "500万下": "1勝",
    "1000万下": "2勝",
    "1600万下": "3勝",
    "2000万下": "3勝",
    "2500万下": "OP",
}

# レース名文字列からグレードを推定する正規表現（優先度順）
_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"G1|ＧⅠ|Ｇ１"), "G1"),
    (re.compile(r"G2|ＧⅡ|Ｇ２"), "G2"),
    (re.compile(r"G3|ＧⅢ|Ｇ３"), "G3"),
    (re.compile(r"新馬"), "新馬"),
    (re.compile(r"未勝利"), "未勝利"),
    (re.compile(r"1勝|500万"), "1勝"),
    (re.compile(r"2勝|1000万"), "2勝"),
    (re.compile(r"3勝|1600万|2000万"), "3勝"),
    (re.compile(r"[（\(]L[）\)]|Listed|リステッド"), "L"),
    (re.compile(r"OP|オープン"), "OP"),
]


def classify_grade(
    grade: str | None,
    race_name: str | None = None,
    race_class: str | None = None,
) -> str:
    """
    grade / race_name / race_class からグレードを正規化して返す。

    優先度:
        1. grade が _GRADE_NORM に定義されている
        2. race_name のパターンマッチ
        3. race_class のパターンマッチ
        4. フォールバック: "OP"（重賞以外の特別競走はOP扱い）
    """
    # 1. grade 直接
    if grade:
        g = str(grade).strip()
        if g in _GRADE_NORM:
            return _GRADE_NORM[g]

    # 2. race_name パターン
    for src in filter(None, [race_name, race_class]):
        for pattern, result in _NAME_PATTERNS:
            if pattern.search(str(src)):
                return result

    return "OP"


def estimate_prize(
    grade: str,
    finish_pos: int | float | None,
) -> float:
    """
    グレードと着順から本賞金（万円）を推計する。

    6着以下 / 入線なし は 0 万円。
    unknown grade は OP として扱う。

    Returns:
        本賞金推計値 (万円, float)
    """
    if finish_pos is None or (isinstance(finish_pos, float) and np.isnan(finish_pos)):
        return 0.0
    pos = int(finish_pos)
    if pos <= 0:
        return 0.0

    table = PRIZE_TABLE.get(grade, PRIZE_TABLE["OP"])
    # 5着以降は PRIZE_TABLE に定義なし → 0
    return float(table.get(min(pos, 5), 0.0)) if pos <= 5 else 0.0


def build_career_prizes_from_flat(
    df_results: pd.DataFrame,
    as_of_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    race_result_flat.parquet を使い、馬ごとの累積獲得賞金を推計する。

    Args:
        df_results: race_result_flat.parquet 相当の DataFrame。
                    必須カラム: horse_id, race_id, date, finish_position,
                               grade, race_name, race_class
        as_of_dates: {horse_id: "YYYYMMDD"} 形式で指定すると
                     その日付より前のレースのみ集計する。
                     None の場合は全レースを集計する。

    Returns:
        DataFrame with columns [horse_id, career_prize_est, last_race_date, n_starts]
    """
    df = df_results.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["finish_pos_num"] = pd.to_numeric(df.get("finish_position", df.get("finish_pos")), errors="coerce")

    # グレード推計
    df["grade_norm"] = df.apply(
        lambda r: classify_grade(
            r.get("grade"),
            r.get("race_name"),
            r.get("race_class"),
        ),
        axis=1,
    )

    # 賞金推計
    df["prize_est"] = df.apply(
        lambda r: estimate_prize(r["grade_norm"], r["finish_pos_num"]),
        axis=1,
    )

    if as_of_dates:
        records = []
        for horse_id, cutoff_str in as_of_dates.items():
            cutoff = pd.to_datetime(cutoff_str)
            sub = df[(df["horse_id"] == horse_id) & (df["date_parsed"] < cutoff)]
            total = sub["prize_est"].sum()
            n = len(sub)
            last_dt = sub["date_parsed"].max() if n > 0 else pd.NaT
            records.append({"horse_id": horse_id, "career_prize_est": total,
                            "n_starts": n, "last_race_date": last_dt})
        return pd.DataFrame(records)

    # 全馬一括集計
    agg = (
        df.groupby("horse_id")
        .agg(
            career_prize_est=("prize_est", "sum"),
            n_starts=("race_id", "count"),
            last_race_date=("date_parsed", "max"),
        )
        .reset_index()
    )
    return agg


def build_career_prizes_from_history(
    horse_histories: dict[str, list[dict]],
    as_of_date: str,
) -> dict[str, float]:
    """
    horse_race_history の raw エントリーリストから
    指定日より前の累積獲得賞金 (万円) を返す。

    Args:
        horse_histories: {horse_id: [race_history_entry, ...]}
            各エントリーのキー: date ("YYYY/MM/DD"), race_name,
                              finish_position (int), grade (省略可)
        as_of_date: "YYYYMMDD" 形式の基準日（この日は含まない）

    Returns:
        {horse_id: career_prize_万円}
    """
    cutoff = pd.to_datetime(as_of_date)
    result: dict[str, float] = {}

    for horse_id, entries in horse_histories.items():
        total = 0.0
        for e in entries:
            raw_date = e.get("date", "")
            try:
                entry_dt = pd.to_datetime(str(raw_date).replace("/", "-"))
            except Exception:
                continue
            if entry_dt >= cutoff:
                continue

            grade = classify_grade(
                e.get("grade"),
                e.get("race_name"),
                e.get("race_class"),
            )
            pos = e.get("finish_position")
            total += estimate_prize(grade, pos)

        result[horse_id] = total

    return result


# ─────────────────────────────────────────────────────────────────────────────
# レベル正規化ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

# 各グレードの1着賞金（level 正規化の基準として使用）
GRADE_1ST_PRIZE: dict[str, float] = {
    k: v[1] for k, v in PRIZE_TABLE.items()
}

# 馬が典型的に経験するレースで獲得する推計中央値 (万円)
# 3勝クラス到達に必要な賞金目安: 未勝利1着 + 1勝1着 + 2勝1着 ≈ 2300万
# （これをグレード level_ratio の denominator として使う）
CAREER_PRIZE_REFERENCE = {
    "新馬":  500,
    "未勝利": 500,
    "1勝":   1500,   # 未勝利 + 1勝 程度
    "2勝":   2500,   # 未勝利 + 1勝 + 2勝 程度
    "3勝":   4000,
    "L":     7000,
    "OP":    7000,
    "G3":   15000,
    "G2":   30000,
    "G1":   80000,
}


def compute_level_feature(
    career_prize: float,
    race_grade: str,
) -> float:
    """
    馬の累積賞金とレースグレードから level フィーチャー (無次元) を計算する。

    定義:
        level_feature = log(career_prize + 1) / log(race_reference + 1) - 1.0

    解釈:
        > 0 → 馬の実績がレース水準より上（格下げ出走）
        = 0 → 馬の実績がレース水準と一致
        < 0 → 馬の実績がレース水準より下（格上挑戦）

    beta_level > 0 の場合、level_feature > 0 の馬は
    delta_level_sec > 0 となり調整後タイムが速くなる（良い成績が期待される）。

    Args:
        career_prize: 累積獲得賞金推計値 (万円, >= 0)
        race_grade:   classify_grade() で正規化されたグレード文字列

    Returns:
        level_feature (float, 無次元)
    """
    ref = CAREER_PRIZE_REFERENCE.get(race_grade, CAREER_PRIZE_REFERENCE["OP"])
    log_career = np.log1p(max(career_prize, 0.0))
    log_ref = np.log1p(ref)
    if log_ref == 0:
        return 0.0
    return float(log_career / log_ref - 1.0)
