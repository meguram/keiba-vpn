"""
特徴量ビルダー

scrape_race_all() の出力（1レース分の全生データ）を受け取り、
予測モデル用の特徴量テーブル (DataFrame) を構築する。

=== 大衆指標排除ポリシー ===
  当日オッズ・人気など市場由来の指標は原則使用しない。
  ただし、前走オッズ・前走人気については「市場評価と実績の乖離」を表す
  派生特徴量（pop_rank_diff, upset_flag 等）に変換して活用する。
  生値（prev{i}_odds, prev{i}_popularity）は派生特徴量生成後に除外する。

特徴量カテゴリ:
  1. 馬の基本属性 (性別, 年齢, 枠番, 馬番, 斤量)
  2. レース条件 (芝/ダート, 距離, 馬場状態, 天候, グレード, 頭数)
  3. 馬体重・増減 (当日計量)
  4. タイム指数 (最高/平均/距離別/コース別/直近3走)
  5. 過去走成績 (直近5走の着順/タイム/上がり3F/着差) — 生オッズ・人気除外
  6. 騎手・調教師統計 (勝率/連対率)
  7. 調教情報 (タイム/ランク/印象) — shutuba_past + oikiri
  8. 血統 (父/母父の種牡馬ID → カテゴリ)
  8b. ミオスタチン遺伝子 (父/母父のアレル確率, 子馬遺伝子型確率, 距離適合度)
  9. 戦績集計 (通算勝率/複勝率, 同距離/同コース成績)
  10. パドック評価 (rank, score)
  11. バロメーター偏差値 (total, start, chase, closing)
  12. 市場乖離派生特徴量 (pop_rank_diff, upset_flag, popularity_trend 等)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 当日オッズ・人気など完全排除対象（前走オッズ・人気は派生特徴量生成後に除外）
PUBLIC_INDICATOR_COLUMNS: list[str] = [
    "odds", "win_odds", "place_odds", "place_odds_min", "place_odds_max",
    "tansho_odds", "fukusho_odds", "tan_odds", "fuku_odds",
    "popularity", "popularity_rank",
    "wakuren_odds", "umaren_odds", "wide_odds", "umatan_odds", "sanren_odds",
    "log_odds", "implied_win_prob", "implied_place_prob", "odds_ratio",
    "market_odds",
]

# 前走オッズ・前走人気は派生特徴量生成後に除外（生値は最終特徴量セットに含めない）
_PREV_RAW_COLUMNS: list[str] = []
for _i in range(1, 11):
    _PREV_RAW_COLUMNS.append(f"prev{_i}_odds")
    _PREV_RAW_COLUMNS.append(f"prev{_i}_popularity")

# 最終除外対象 = 当日指標 + 前走生値
PUBLIC_INDICATOR_COLUMNS = PUBLIC_INDICATOR_COLUMNS + _PREV_RAW_COLUMNS

PUBLIC_INDICATOR_SET = frozenset(PUBLIC_INDICATOR_COLUMNS)

# 後方互換エイリアス
ODDS_RELATED_COLUMNS = PUBLIC_INDICATOR_COLUMNS


def _drop_public_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if c in PUBLIC_INDICATOR_SET]
    if cols_to_drop:
        logger.info("大衆指標カラムを除外: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


def _filter_public_indicators(feature_cols: list[str]) -> list[str]:
    excluded = [c for c in feature_cols if c in PUBLIC_INDICATOR_SET]
    if excluded:
        logger.info("大衆指標カラムをリストから除外: %s", excluded)
    return [c for c in feature_cols if c not in PUBLIC_INDICATOR_SET]


# 後方互換エイリアス
_drop_odds_columns = _drop_public_indicators
_filter_odds_columns = _filter_public_indicators


# ─── 市場乖離派生特徴量 ──────────────────────────────────


def build_past_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    前走オッズ・人気の生値から「市場評価と実績の乖離」を表す派生特徴量を生成する。

    生成する特徴量:
      - prev{i}_pop_rank_diff  : popularity - finish_position（正=好走、負=凡走）
      - prev{i}_upset_flag     : 人気5以下かつ3着以内なら1（穴好走フラグ）
      - avg_prev_pop_rank_diff : prev1〜5の pop_rank_diff 平均
      - popularity_trend_slope : prev1〜5の popularity の線形回帰傾き
      - upset_count            : prev1〜5の upset_flag 合計
      - prev1_odds_vs_current_ratio : prev1_odds / odds（市場評価変化率）

    Parameters
    ----------
    df : pd.DataFrame
        build_race_features() が生成した生特徴量テーブル（前走生値を含む状態）

    Returns
    -------
    pd.DataFrame
        派生特徴量のみを含む DataFrame（行数・インデックスは df と同一）
        生の prev{i}_odds・prev{i}_popularity は含まない。
    """
    n = len(df)
    result: dict[str, list] = {}

    # ── 1. prev{i}_pop_rank_diff & prev{i}_upset_flag ──────────────
    for i in range(1, 6):
        pop_col = f"prev{i}_popularity"
        fin_col = f"prev{i}_finish"
        diff_col = f"prev{i}_pop_rank_diff"
        flag_col = f"prev{i}_upset_flag"

        diff_vals: list[float] = []
        flag_vals: list[float] = []

        for idx in range(n):
            pop = _get_numeric(df, idx, pop_col)
            fin = _get_numeric(df, idx, fin_col)

            # finish_position が 0 や欠損の場合は計算対象外
            if pop is None or fin is None or fin <= 0:
                diff_vals.append(float("nan"))
                flag_vals.append(float("nan"))
            else:
                diff_vals.append(pop - fin)
                flag_vals.append(1.0 if (pop >= 5 and fin <= 3) else 0.0)

        result[diff_col] = diff_vals
        result[flag_col] = flag_vals

    # ── 2. avg_prev_pop_rank_diff ──────────────────────────────────
    avg_diffs: list[float] = []
    for idx in range(n):
        vals = []
        for i in range(1, 6):
            v = result[f"prev{i}_pop_rank_diff"][idx]
            if not math.isnan(v):
                vals.append(v)
        avg_diffs.append(sum(vals) / len(vals) if vals else float("nan"))
    result["avg_prev_pop_rank_diff"] = avg_diffs

    # ── 3. popularity_trend_slope ──────────────────────────────────
    trend_slopes: list[float] = []
    for idx in range(n):
        pops = []
        xs = []
        for i in range(1, 6):
            pop_col = f"prev{i}_popularity"
            v = _get_numeric(df, idx, pop_col)
            if v is not None and not math.isnan(v):
                pops.append(v)
                xs.append(i)  # 1=直近, 5=5走前

        if len(pops) < 2:
            trend_slopes.append(float("nan"))
        else:
            try:
                slope = float(np.polyfit(xs, pops, 1)[0])
                trend_slopes.append(slope)
            except (np.linalg.LinAlgError, ValueError):
                trend_slopes.append(float("nan"))
    result["popularity_trend_slope"] = trend_slopes

    # ── 4. upset_count ────────────────────────────────────────────
    upset_counts: list[float] = []
    for idx in range(n):
        total = 0.0
        has_data = False
        for i in range(1, 6):
            v = result[f"prev{i}_upset_flag"][idx]
            if not math.isnan(v):
                total += v
                has_data = True
        upset_counts.append(total if has_data else float("nan"))
    result["upset_count"] = upset_counts

    # ── 5. prev1_odds_vs_current_ratio ────────────────────────────
    ratio_vals: list[float] = []
    for idx in range(n):
        prev1_odds = _get_numeric(df, idx, "prev1_odds")
        current_odds = _get_numeric(df, idx, "odds")

        if (prev1_odds is None or math.isnan(prev1_odds)
                or current_odds is None or math.isnan(current_odds)
                or current_odds == 0.0):
            ratio_vals.append(float("nan"))
        else:
            ratio_vals.append(prev1_odds / current_odds)
    result["prev1_odds_vs_current_ratio"] = ratio_vals

    derived_df = pd.DataFrame(result, index=df.index)
    # float 型を明示的に保持
    for col in derived_df.columns:
        derived_df[col] = derived_df[col].astype(float)

    return derived_df


def _get_numeric(df: pd.DataFrame, idx: int, col: str) -> float | None:
    """DataFrame の指定行・列から数値を安全に取得する。欠損・非数値は None を返す。"""
    if col not in df.columns:
        return None
    try:
        val = df.iloc[idx][col]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        f = float(val)
        return f
    except (ValueError, TypeError):
        return None


# ─── メイン特徴量ビルダー ────────────────────────────────


def build_race_features(race_data: dict) -> pd.DataFrame:
    """
    scrape_race_all() の出力を受け取り、1レース分の特徴量テーブルを返す。
    各行 = 1頭。列 = 特徴量。

    処理順序:
      1. 生特徴量を構築（前走オッズ・人気の生値を含む）
      2. build_past_market_features() で派生特徴量を生成・結合
      3. _drop_public_indicators() で生値（前走オッズ・人気含む）を除外
    """
    card = race_data.get("race_card") or {}
    speed = race_data.get("speed_index") or {}
    past = race_data.get("shutuba_past") or {}
    horses = race_data.get("horses") or {}
    paddock = race_data.get("paddock") or {}
    barometer = race_data.get("barometer") or {}
    oikiri = race_data.get("oikiri") or {}

    entries = card.get("entries", [])
    if not entries:
        return pd.DataFrame()

    speed_map = {e["horse_number"]: e for e in speed.get("entries", [])}
    past_map = {e["horse_number"]: e for e in past.get("entries", [])}
    paddock_map = {e["horse_number"]: e for e in paddock.get("entries", [])}
    barometer_map = {e["horse_number"]: e for e in barometer.get("entries", [])}
    oikiri_map = {e["horse_number"]: e for e in oikiri.get("entries", [])}

    training_list = past.get("training", [])
    training_map: dict[int, list] = {}
    for t in training_list:
        hn = t.get("horse_number", 0)
        training_map.setdefault(hn, []).append(t)

    race_info = _extract_race_info(card)
    rows = []

    for entry in entries:
        hn = entry.get("horse_number", 0)
        hid = entry.get("horse_id", "")
        horse_profile = horses.get(hid, {})
        horse_history = horse_profile.get("race_history", [])

        row: dict[str, Any] = {}

        row["race_id"] = race_data.get("race_id", "")
        row["horse_number"] = hn
        row["horse_name"] = entry.get("horse_name", "")
        row["horse_id"] = hid

        row.update(race_info)
        row.update(_horse_basics(entry))
        row.update(_speed_features(speed_map.get(hn, {})))
        row.update(_past_race_features(past_map.get(hn, {}), horse_history))
        row.update(_training_features(
            training_map.get(hn, []),
            oikiri_map.get(hn, {}),
        ))
        row.update(_career_stats(horse_history, race_info))
        row.update(_pedigree_features(entry, horse_profile, race_info.get("distance", 0)))
        row.update(_paddock_features(paddock_map.get(hn, {})))
        row.update(_barometer_features(barometer_map.get(hn, {})))

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── 派生特徴量を生成してから生値を除外 ──
    derived = build_past_market_features(df)
    df = pd.concat([df, derived], axis=1)
    df = _drop_public_indicators(df)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return _filter_public_indicators(list(df.columns))


def prepare_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = list(df.columns)
    feature_cols = _filter_public_indicators(feature_cols)
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.info("prepare_features: 存在しないカラムをスキップ: %s", missing)
    return df[available_cols]


# ─── 個別特徴量関数 ──────────────────────────────────


def _extract_race_info(card: dict) -> dict:
    return {
        "venue": card.get("venue", ""),
        "surface": card.get("surface", ""),
        "distance": card.get("distance", 0),
        "direction": card.get("direction", ""),
        "weather": card.get("weather", ""),
        "track_condition": card.get("track_condition", ""),
        "field_size": card.get("field_size", 0) or len(card.get("entries", [])),
        "race_round": card.get("round", 0),
        "race_date": card.get("date", ""),
    }


def _horse_basics(entry: dict) -> dict:
    sex_age = entry.get("sex_age", "")
    sex, age = "", 0
    m = re.match(r"(牡|牝|セ)(\d+)", sex_age)
    if m:
        sex, age = m.group(1), int(m.group(2))

    return {
        "bracket_number": entry.get("bracket_number", 0),
        "sex": sex,
        "age": age,
        "jockey_weight": entry.get("jockey_weight", 0),
        "jockey_name": entry.get("jockey_name", ""),
        "trainer_name": entry.get("trainer_name", ""),
        "weight": entry.get("weight", 0),
        "weight_change": entry.get("weight_change", 0),
    }


def _speed_features(sp: dict) -> dict:
    recent = sp.get("speed_recent") or [0, 0, 0]
    return {
        "speed_max": sp.get("speed_max", 0),
        "speed_avg": sp.get("speed_avg", 0),
        "speed_distance": sp.get("speed_distance", 0),
        "speed_course": sp.get("speed_course", 0),
        "speed_recent_1": recent[0] if len(recent) > 0 else 0,
        "speed_recent_2": recent[1] if len(recent) > 1 else 0,
        "speed_recent_3": recent[2] if len(recent) > 2 else 0,
    }


def _past_race_features(past_entry: dict, horse_history: list[dict]) -> dict:
    """直近5走の成績を特徴量化。前走オッズ・人気の生値も一時的に保持（派生特徴量生成後に除外）。"""
    features: dict[str, Any] = {}
    recent = horse_history[:5] if horse_history else []

    for i in range(5):
        prefix = f"prev{i+1}_"

        if i < len(recent):
            r = recent[i]
            features[prefix + "finish"] = r.get("finish_position", 0)
            features[prefix + "last_3f"] = r.get("last_3f", 0)
            features[prefix + "weight"] = r.get("weight", 0)
            features[prefix + "weight_change"] = r.get("weight_change", 0)
            features[prefix + "distance"] = r.get("distance", 0)
            features[prefix + "surface"] = r.get("surface", "")
            features[prefix + "track_cond"] = r.get("track_condition", "")

            time_str = r.get("finish_time", "")
            features[prefix + "time_sec"] = _time_to_sec(time_str)

            passing = r.get("passing_order", "")
            positions = re.findall(r"\d+", passing)
            features[prefix + "pass_first"] = int(positions[0]) if positions else 0
            features[prefix + "pass_last"] = int(positions[-1]) if positions else 0

            features[prefix + "field_size"] = r.get("field_size", 0)
            features[prefix + "finish_diff"] = _finish_time_diff(r)

            # 前走オッズ・人気は派生特徴量生成のために一時保持（後段で除外）
            features[prefix + "odds"] = _safe_float_or_nan(r.get("odds"))
            features[prefix + "popularity"] = _safe_float_or_nan(r.get("popularity"))
        else:
            for col in ("finish", "last_3f", "weight", "weight_change",
                        "distance", "time_sec", "pass_first", "pass_last",
                        "field_size", "finish_diff"):
                features[prefix + col] = 0
            features[prefix + "surface"] = ""
            features[prefix + "track_cond"] = ""
            # 前走データなし → NaN で保持
            features[prefix + "odds"] = float("nan")
            features[prefix + "popularity"] = float("nan")

    if recent:
        finishes = [r.get("finish_position", 0) for r in recent
                    if r.get("finish_position", 0) > 0]
        features["avg_finish_5"] = sum(finishes) / len(finishes) if finishes else 0
        features["min_finish_5"] = min(finishes) if finishes else 0
        features["win_count_5"] = sum(1 for f in finishes if f == 1)
        features["top3_count_5"] = sum(1 for f in finishes if 1 <= f <= 3)

        last_3fs = [r.get("last_3f", 0) for r in recent if r.get("last_3f", 0) > 0]
        features["avg_last_3f_5"] = sum(last_3fs) / len(last_3fs) if last_3fs else 0
        features["min_last_3f_5"] = min(last_3fs) if last_3fs else 0
        features["std_last_3f_5"] = _std(last_3fs)

        time_secs = [_time_to_sec(r.get("finish_time", "")) for r in recent]
        time_secs = [t for t in time_secs if t > 0]
        features["avg_time_sec_5"] = sum(time_secs) / len(time_secs) if time_secs else 0

        intervals = _calc_race_intervals(recent)
        features["days_since_last"] = intervals[0] if intervals else 0
        features["avg_interval"] = sum(intervals) / len(intervals) if intervals else 0

        features["position_trend"] = _position_trend(finishes)
        features["last_3f_trend"] = _trend(last_3fs)
    else:
        for col in ("avg_finish_5", "min_finish_5", "win_count_5", "top3_count_5",
                     "avg_last_3f_5", "min_last_3f_5", "std_last_3f_5",
                     "avg_time_sec_5", "days_since_last", "avg_interval",
                     "position_trend", "last_3f_trend"):
            features[col] = 0

    return features


def _training_features(trainings: list[dict], oikiri_entry: dict) -> dict:
    """shutuba_past の調教 + oikiri のデータを統合した調教特徴量。"""
    features: dict[str, Any] = {
        "training_count": 0,
        "best_training_rank": 0,
        "training_time_4f": 0.0,
        "training_impression_score": 0,
        "oikiri_evaluation": 0,
        "oikiri_lap_best": 0.0,
        "oikiri_lap_count": 0,
    }

    impression_scores = {
        "馬なり": 1, "強め": 2, "仕掛け": 2, "一杯": 3,
        "末強め": 2, "G前仕掛け": 2, "直強め": 2,
    }
    evaluation_scores = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

    if trainings:
        features["training_count"] = len(trainings)
        ranks = [int(t.get("rank", 0) or 0) for t in trainings if t.get("rank")]
        imp_scores = [impression_scores.get(t.get("impression", ""), 0) for t in trainings]

        features["best_training_rank"] = min(ranks) if ranks else 0
        features["training_impression_score"] = max(imp_scores) if imp_scores else 0

        fastest_4f = 0.0
        for t in trainings:
            time_str = t.get("time", "")
            parts = time_str.split("-")
            if parts:
                try:
                    fastest_4f = max(fastest_4f, float(parts[0]))
                except ValueError:
                    pass
        features["training_time_4f"] = fastest_4f

    if oikiri_entry:
        eval_str = oikiri_entry.get("evaluation", "")
        features["oikiri_evaluation"] = evaluation_scores.get(eval_str, 0)

        laps = oikiri_entry.get("lap_times", [])
        if isinstance(laps, list) and laps:
            valid_laps = []
            for l in laps:
                try:
                    valid_laps.append(float(l))
                except (ValueError, TypeError):
                    pass
            if valid_laps:
                features["oikiri_lap_best"] = min(valid_laps)
                features["oikiri_lap_count"] = len(valid_laps)

        imp = oikiri_entry.get("impression", "")
        if imp and not trainings:
            features["training_impression_score"] = impression_scores.get(imp, 0)

    return features


def _career_stats(history: list[dict], race_info: dict) -> dict:
    if not history:
        return {
            "career_runs": 0, "career_win_rate": 0.0, "career_top3_rate": 0.0,
            "same_surface_runs": 0, "same_surface_win_rate": 0.0,
            "same_dist_runs": 0, "same_dist_win_rate": 0.0,
            "same_venue_runs": 0, "same_venue_win_rate": 0.0,
            "career_avg_finish": 0.0, "career_avg_last_3f": 0.0,
        }

    total = len(history)
    valid = [h for h in history if h.get("finish_position", 0) > 0]
    wins = sum(1 for h in valid if h["finish_position"] == 1)
    top3 = sum(1 for h in valid if 1 <= h["finish_position"] <= 3)

    target_surface = race_info.get("surface", "")[:1]
    target_dist = race_info.get("distance", 0)
    target_venue = race_info.get("venue", "")

    same_surface = [h for h in valid if (h.get("surface", "")[:1] or "") == target_surface]
    same_dist = [h for h in valid if abs(h.get("distance", 0) - target_dist) <= 200]
    same_venue = [h for h in valid if h.get("venue", "") == target_venue]

    def _wr(subset):
        return sum(1 for h in subset if h["finish_position"] == 1) / len(subset) if subset else 0.0

    avg_finish = (sum(h["finish_position"] for h in valid) / len(valid)) if valid else 0.0
    last_3fs = [h.get("last_3f", 0) for h in valid if h.get("last_3f", 0) > 0]
    avg_last_3f = sum(last_3fs) / len(last_3fs) if last_3fs else 0.0

    return {
        "career_runs": total,
        "career_win_rate": wins / len(valid) if valid else 0.0,
        "career_top3_rate": top3 / len(valid) if valid else 0.0,
        "same_surface_runs": len(same_surface),
        "same_surface_win_rate": _wr(same_surface),
        "same_dist_runs": len(same_dist),
        "same_dist_win_rate": _wr(same_dist),
        "same_venue_runs": len(same_venue),
        "same_venue_win_rate": _wr(same_venue),
        "career_avg_finish": avg_finish,
        "career_avg_last_3f": avg_last_3f,
    }


def _pedigree_features(entry: dict, profile: dict, distance: int = 0) -> dict:
    sire = entry.get("sire", "") or profile.get("sire", "")
    dam_sire = entry.get("dam_sire", "") or profile.get("dam_sire", "")
    broodmare_line = (
        (entry.get("broodmare_line") or profile.get("broodmare_line") or "")
        .strip()
        or None
    )

    feats: dict[str, Any] = {"sire": sire, "dam_sire": dam_sire}

    try:
        from research.myostatin import get_lookup
        mstn = get_lookup()
        feats.update(mstn.offspring_features(sire, dam_sire, distance))
    except Exception:
        feats.update({
            "sire_mstn_c": 0.0, "sire_mstn_t": 0.0,
            "dam_sire_mstn_c": 0.0, "dam_sire_mstn_t": 0.0,
            "mstn_cc_prob": 0.0, "mstn_ct_prob": 0.0, "mstn_tt_prob": 0.0,
            "mstn_speed_index": 0.0, "mstn_distance_affinity": 0.0,
        })

    try:
        from research.sire_aptitude_note import compute_note_aptitude_features

        feats.update(
            compute_note_aptitude_features(
                sire, dam_sire, distance, broodmare_line=broodmare_line
            )
        )
    except Exception:
        try:
            from research.sire_aptitude_note import zero_note_aptitude_features

            feats.update(zero_note_aptitude_features())
        except Exception:
            feats.update({"note_apt_dist_fit": 0.0, "note_apt_l2": 0.0})

    return feats


def _paddock_features(paddock_entry: dict) -> dict:
    """パドック評価。数値化されたランクがあればそのまま使用。"""
    if not paddock_entry:
        return {"paddock_rank": 0, "paddock_score": 0.0}

    rank = paddock_entry.get("paddock_rank", 0)
    try:
        rank = int(rank)
    except (ValueError, TypeError):
        rank = 0

    comment = paddock_entry.get("paddock_comment", "")
    score = _paddock_comment_to_score(comment) if comment else 0.0

    return {"paddock_rank": rank, "paddock_score": score}


def _barometer_features(barometer_entry: dict) -> dict:
    """バロメーター偏差値。index_total 等は netkeiba が付与する偏差値指標。"""
    if not barometer_entry:
        return {
            "barometer_total": 0.0,
            "barometer_start": 0.0,
            "barometer_chase": 0.0,
            "barometer_closing": 0.0,
        }

    return {
        "barometer_total": _safe_float(barometer_entry.get("index_total", 0)),
        "barometer_start": _safe_float(barometer_entry.get("index_start", 0)),
        "barometer_chase": _safe_float(barometer_entry.get("index_chase", 0)),
        "barometer_closing": _safe_float(barometer_entry.get("index_closing", 0)),
    }


# ─── ユーティリティ ──────────────────────────────────


def _time_to_sec(t: str) -> float:
    if not t:
        return 0.0
    m = re.match(r"(\d+):(\d+\.\d+)", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m2 = re.match(r"(\d+\.\d+)", t)
    return float(m2.group(1)) if m2 else 0.0


def _safe_float(x) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def _safe_float_or_nan(x) -> float:
    """変換できない場合は NaN を返す（0 ではなく欠損として扱う）。"""
    if x is None:
        return float("nan")
    try:
        f = float(x)
        return f
    except (ValueError, TypeError):
        return float("nan")


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def _trend(vals: list[float]) -> float:
    """値リストの傾向 (負=改善, 正=悪化)。最小二乗法の傾き。"""
    if len(vals) < 2:
        return 0.0
    n = len(vals)
    x_mean = (n - 1) / 2
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _position_trend(finishes: list[int]) -> float:
    """着順の改善傾向。負=上昇傾向。"""
    if len(finishes) < 2:
        return 0.0
    return _trend([float(f) for f in finishes])


def _finish_time_diff(race_record: dict) -> float:
    """着差を秒に変換。先頭からの差（大きいほど離された）。"""
    margin = race_record.get("margin", "")
    if not margin or margin in ("", "-"):
        return 0.0
    margin_map = {
        "ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2, "1/2": 0.3,
        "3/4": 0.45, "1": 0.6, "1 1/4": 0.75, "1 1/2": 0.9,
        "1 3/4": 1.05, "2": 1.2, "2 1/2": 1.5, "3": 1.8,
        "大差": 5.0,
    }
    return margin_map.get(margin.strip(), 0.0)


def _paddock_comment_to_score(comment: str) -> float:
    """パドックコメントから状態スコアを推定。"""
    positive = ["好", "良", "活気", "元気", "落ち着", "気合", "前向き", "馬体増"]
    negative = ["太", "細", "入れ込", "発汗", "テンション高", "チャカ", "うるさ"]
    score = 0.0
    for w in positive:
        if w in comment:
            score += 1.0
    for w in negative:
        if w in comment:
            score -= 1.0
    return score


def _calc_race_intervals(history: list[dict]) -> list[int]:
    from datetime import datetime
    dates = []
    for h in history:
        d = h.get("date", "")
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d"):
            try:
                dates.append(datetime.strptime(d.strip(), fmt))
                break
            except ValueError:
                continue
    intervals = []
    for i in range(len(dates) - 1):
        delta = abs((dates[i] - dates[i + 1]).days)
        intervals.append(delta)
    return intervals
