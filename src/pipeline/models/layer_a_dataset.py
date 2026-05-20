"""
LayerA 学習母表の構築（BaselineV1 相当・当日レース直前パイプラインと整合）

- 特徴: race_shutuba_flat + race_index_flat（指数のみ結合）+ race_shutuba_past_flat
- 除外: 当日 win_odds / popularity（shutuba・index）および PUBLIC_INDICATOR_SET に含まれる列
- ラベル: race_result_flat.finish_position（学習時のみ結合。推論パイプラインでは使用しない）
- 任意: 馬場傾向×血統 preprocess（pipeline/track_bias_pedigree.py）

参照: docs/html/modeling/pre_race_feature_whitelist.html（BaselineV1）
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.features.feature_builder import _drop_public_indicators
from src.pipeline.features.race_index_speed_recent import expand_speed_recent_in_dataframe

logger = logging.getLogger(__name__)

# 指数テーブルから結合する列（レース条件の重複は shutuba 側を正とする）
# time_index_m はパーサ上の推定値で 0 欠損と区別しづらいため除外。
# speed_recent は結合後に speed_recent_1..3 に展開する。
INDEX_FEATURE_COLS = [
    "race_id",
    "horse_number",
    "speed_max",
    "speed_avg",
    "speed_distance",
    "speed_course",
    "speed_recent",
    "all_txt_c",
]

PAST_FEATURE_COLS = [
    "race_id",
    "horse_number",
    "past_races",
    "training",
]

RESULT_LABEL_COLS = ["race_id", "horse_number", "finish_position"]

DEFAULT_OUTPUT_PARQUET = Path("data/local/modeling/layer_a_train.parquet")
DEFAULT_META_JSON = Path("data/local/modeling/layer_a_train.meta.json")


def _jra_mask(df: pd.DataFrame) -> pd.Series:
    """JRA 開催のみ（venue_code 01-10 = 中央10場）。データに無ければ全 True。"""
    if "venue_code" not in df.columns:
        return pd.Series(True, index=df.index)
    vc = pd.to_numeric(df["venue_code"], errors="coerce")
    return vc.between(1, 10, inclusive="both")


def build_layer_a_dataframe(
    *,
    years: list[str | int] | None = None,
    jra_only: bool = True,
    base_dir: str | Path = ".",
    include_history_stats: bool = True,
    include_jockey_trainer_stats: bool = False,
    include_track_bias_pedigree: bool = True,
    track_bias_same_day_weight: float | None = None,
    track_bias_prev_day_weight: float | None = None,
    track_bias_weights_json: str | Path | None = None,
) -> pd.DataFrame:
    """
    data/local/tables の flat Parquet から LayerA 母表を構築する。

    include_history_stats:
      race_result 全期間を用い、馬・騎手・調教師のリークなし累計（勝率・複勝率・平均着など）を付与。
      preprocess: pipeline/historical_entity_stats.py
    include_jockey_trainer_stats:
      ``data/features/race_horse_tbl/<YYYY>/jt_race_features.parquet`` を結合（事前に
      ``python -m pipeline.build_jockey_trainer_stats`` が必要）。主キーは ``race_id`` + ``horse_id``。
    include_track_bias_pedigree:
      当日・前日の芝/ダ別馬場傾向（勝者上がり3F）と父・母父の芝/ダ別累計勝率・交互作用。
      preprocess: pipeline/track_bias_pedigree.py（race_result 全利用年 + race_shutuba の血統列）
    track_bias_same_day_weight / track_bias_prev_day_weight:
      当日（該当レース前）と前日の傾向の合成重み（None 時は track_bias_pedigree の既定 0.75 / 0.25）。
    track_bias_weights_json:
      tune_track_bias_weights の出力 JSON パス。指定時は上記 float より優先。
    """
    from src.pipeline.features.feature_store import FeatureStore

    store = FeatureStore(base_dir=base_dir)
    avail = store.available_years()
    if not avail:
        raise FileNotFoundError("data/local/tables/<年>/ に Parquet がありません。")
    use_years = [str(y) for y in years] if years else avail
    use_years = [y for y in use_years if y in avail]
    if not use_years:
        raise ValueError(f"指定年 {years} にデータがありません。利用可能: {avail}")

    logger.info("LayerA 構築: years=%s", use_years)

    sb = store.load_source("race_shutuba", years=use_years)
    if sb.empty:
        raise RuntimeError("race_shutuba が空です")

    drop_mkt = {"odds", "popularity"}
    sb = sb.drop(columns=[c for c in drop_mkt if c in sb.columns], errors="ignore")

    ix_cols = [c for c in INDEX_FEATURE_COLS if c in store.source_schema("race_index", use_years[0])]
    ix = store.load_source("race_index", years=use_years, columns=ix_cols)
    ix = ix.drop(columns=[c for c in drop_mkt if c in ix.columns], errors="ignore")
    ix = expand_speed_recent_in_dataframe(ix, "speed_recent", dest_prefix="speed_recent")

    past_cols = [
        c
        for c in PAST_FEATURE_COLS
        if c in store.source_schema("race_shutuba_past", use_years[0])
    ]
    past = store.load_source("race_shutuba_past", years=use_years, columns=past_cols)

    m = sb.merge(ix, on=["race_id", "horse_number"], how="inner")
    m = m.merge(past, on=["race_id", "horse_number"], how="left")

    if jra_only:
        keep = _jra_mask(m)
        n_drop = int((~keep).sum())
        if n_drop:
            logger.info("JRA 以外を除外: %d 行", n_drop)
        m = m.loc[keep].copy()

    rr = store.load_source("race_result", years=use_years, columns=RESULT_LABEL_COLS)
    if rr.empty:
        raise RuntimeError("race_result が空です（ラベル結合不可）")

    m = m.merge(rr, on=["race_id", "horse_number"], how="inner")
    m = m.drop_duplicates(subset=["race_id", "horse_number"], keep="first")
    m["finish_position"] = pd.to_numeric(m["finish_position"], errors="coerce")
    m = m[m["finish_position"].notna() & (m["finish_position"] > 0)]

    m = _drop_public_indicators(m)

    # 長文テキストはベースラインでは学習に入れない（数値化されていないため）
    if "all_txt_c" in m.columns:
        m = m.drop(columns=["all_txt_c"])
        logger.info("all_txt_c を除外（ベースライン数値化前）")

    if include_history_stats:
        from src.pipeline.features.historical_entity_stats import merge_history_into_layer_a

        hist_cols = [
            "race_id",
            "horse_number",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "date",
            "finish_position",
        ]
        rr_hist = store.load_source("race_result", years=avail, columns=hist_cols)
        if not rr_hist.empty:
            m = merge_history_into_layer_a(m, rr_hist)
            logger.info(
                "履歴統計を付与（race_result 全利用年=%s, 行=%d）",
                avail,
                len(rr_hist),
            )

    if include_jockey_trainer_stats:
        from src.pipeline.features.jockey_trainer_stats import merge_jt_race_features_into_layer_a

        m = merge_jt_race_features_into_layer_a(m, base_dir=base_dir)
        logger.info("騎手・調教師拡張統計を付与（jt_race_features）")

    if include_track_bias_pedigree:
        from src.pipeline.features.track_bias_pedigree import (
            DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT,
            DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT,
            load_track_bias_weights_config,
            merge_track_bias_pedigree_into_layer_a,
        )

        tb_cm: dict[str, float] | None = None
        tb_dm: dict[str, float] | None = None
        if track_bias_weights_json:
            jp = Path(track_bias_weights_json)
            if not jp.is_file():
                raise FileNotFoundError(f"track_bias_weights_json が見つかりません: {jp}")
            cfg = load_track_bias_weights_config(jp)
            tb_sw = float(cfg["best_same_day_weight"])
            tb_pw = float(cfg["best_prev_day_weight"])
            tb_cm = cfg["cond_multipliers"]
            tb_dm = cfg["dist_multipliers"]
        else:
            tb_sw = (
                track_bias_same_day_weight
                if track_bias_same_day_weight is not None
                else DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT
            )
            tb_pw = (
                track_bias_prev_day_weight
                if track_bias_prev_day_weight is not None
                else DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT
            )

        tb_cols = [
            c
            for c in (
                "race_id",
                "horse_number",
                "date",
                "venue_code",
                "surface",
                "start_time",
                "finish_position",
                "last_3f",
                "track_condition",
                "distance",
            )
            if c in store.source_schema("race_result", use_years[0])
        ]
        rr_tb = store.load_source("race_result", years=avail, columns=tb_cols)
        sb_cols = [
            c
            for c in ("race_id", "horse_number", "sire", "dam_sire")
            if c in store.source_schema("race_shutuba", use_years[0])
        ]
        sb_tb = store.load_source("race_shutuba", years=avail, columns=sb_cols)
        if jra_only and not rr_tb.empty:
            keep_tb = _jra_mask(rr_tb)
            rr_tb = rr_tb.loc[keep_tb].copy()
        if not rr_tb.empty and not sb_tb.empty:
            m = merge_track_bias_pedigree_into_layer_a(
                m,
                rr_tb,
                sb_tb,
                same_day_weight=tb_sw,
                prev_day_weight=tb_pw,
                cond_multipliers=tb_cm,
                dist_multipliers=tb_dm,
            )
            logger.info(
                "馬場傾向×血統特徴を付与（race_result 全利用年=%s, rr_tb 行=%d, "
                "bias 重み same_day=%.3f prev_day=%.3f）",
                avail,
                len(rr_tb),
                tb_sw,
                tb_pw,
            )

    return m.reset_index(drop=True)


def write_layer_a_dataset(
    *,
    years: list[str | int] | None = None,
    jra_only: bool = True,
    include_history_stats: bool = True,
    include_jockey_trainer_stats: bool = False,
    include_track_bias_pedigree: bool = True,
    track_bias_same_day_weight: float | None = None,
    track_bias_prev_day_weight: float | None = None,
    track_bias_weights_json: str | Path | None = None,
    output_parquet: Path | None = None,
    meta_path: Path | None = None,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """母表を Parquet + メタ JSON に書き出す。"""
    out = Path(output_parquet or DEFAULT_OUTPUT_PARQUET)
    meta_p = Path(meta_path or DEFAULT_META_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)

    from src.pipeline.features.track_bias_pedigree import (
        DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT,
        DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT,
        load_track_bias_weights_config,
    )

    eff_tb_cm: dict[str, float] | None = None
    eff_tb_dm: dict[str, float] | None = None
    if track_bias_weights_json:
        cfg = load_track_bias_weights_config(Path(track_bias_weights_json))
        eff_tb_sw = float(cfg["best_same_day_weight"])
        eff_tb_pw = float(cfg["best_prev_day_weight"])
        eff_tb_cm = cfg["cond_multipliers"]
        eff_tb_dm = cfg["dist_multipliers"]
    else:
        eff_tb_sw = (
            track_bias_same_day_weight
            if track_bias_same_day_weight is not None
            else DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT
        )
        eff_tb_pw = (
            track_bias_prev_day_weight
            if track_bias_prev_day_weight is not None
            else DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT
        )

    df = build_layer_a_dataframe(
        years=years,
        jra_only=jra_only,
        base_dir=base_dir,
        include_history_stats=include_history_stats,
        include_jockey_trainer_stats=include_jockey_trainer_stats,
        include_track_bias_pedigree=include_track_bias_pedigree,
        track_bias_same_day_weight=track_bias_same_day_weight,
        track_bias_prev_day_weight=track_bias_prev_day_weight,
        track_bias_weights_json=Path(track_bias_weights_json)
        if track_bias_weights_json
        else None,
    )
    df.to_parquet(out, index=False)

    schema = "layer_a_baseline_v1"
    if include_history_stats:
        schema = "layer_a_baseline_v2_with_history"
    if include_jockey_trainer_stats:
        schema = f"{schema}_jt_stats"
    if include_track_bias_pedigree:
        schema = f"{schema}_track_bias_pedigree"

    info = {
        "schema": schema,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_parquet": str(out),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "years": sorted(df["race_id"].astype(str).str[:4].unique().tolist()),
        "jra_only": jra_only,
        "include_history_stats": include_history_stats,
        "include_jockey_trainer_stats": include_jockey_trainer_stats,
        "include_track_bias_pedigree": include_track_bias_pedigree,
        "track_bias_same_day_weight": eff_tb_sw,
        "track_bias_prev_day_weight": eff_tb_pw,
        "track_bias_cond_multipliers": eff_tb_cm,
        "track_bias_dist_multipliers": eff_tb_dm,
        "track_bias_weights_json": str(track_bias_weights_json)
        if track_bias_weights_json
        else None,
        "notes": [
            "当日 odds/popularity 除外済み（race_shutuba / race_index）",
            "ラベルは race_result.finish_position のみ結合",
            "推論時は race_result を結合しない",
            "履歴統計: race_result 全利用年で時系列累計（当レースより過去のみ）。推論時は同一ロジックで当日以前のみ参照",
            "馬場傾向×血統: 芝/ダ別の当日（当該レース前）・前日の勝者上がり3Fと父・母父の芝/ダ別累計勝率（当該レースより過去のみ）",
            "騎手・調教師拡張統計（任意）: data/features/race_horse_tbl/<YYYY>/jt_race_features.parquet を結合",
            "加重馬場傾向: track_bias_winner_3f_weighted = w_same×当日該当レース前 + w_prev×前日（既定 w_same=0.75, w_prev=0.25）。相性度は sire_x_track_bias_weighted 等",
        ],
    }
    meta_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("書き出し: %s (%d 行)", out, len(df))
    return info
