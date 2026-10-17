"""推論時に LayerA 由来の 48 列（レース条件・出走情報・騎手/調教師/馬の通算成績・馬場傾向・父/母父の勝率）を作る。

学習側（``layer_a_dataset.build_layer_a_dataframe`` と ``load_table``）と同じ定義で、結果が未確定の出走馬（出馬表）に付ける。

方式: 学習と同じ関数・同じ並び順を使い、「履歴（結果あり）＋対象レース（結果なし）」を連結して計算し、対象行だけ返す。
- 通算成績（``historical_entity_stats`` と同じ関数 ``strict_prior_by_date``）: 日付が厳密に前の履歴の行だけを数える（同日・同一レースの結果は数えない）。
- 馬場傾向・父/母父の勝率（``track_bias_pedigree.build_track_bias_pedigree_features``）: 結果のないレースにプレースホルダ行を足して
  傾向を付ける。未確定レースは同日の後のレースの「当日直前傾向」に混ざらない。

注意（学習側の仕様）:
- 出馬表の sire / dam_sire は全期間で空のため、学習表の ``sire_prior_win_rate_surface`` などは「面ごとの全体の累計勝率」になっている。
  推論でも同じ（空のまま扱う）。血統名が入るようになったら学習表の作り直しが先。
- 学習側の通算成績も 2026-10-18 から『日付が厳密に前』に直したので、推論と同じ定義（以前は同一レースの他馬の結果まで数える先読みがあった）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features.historical_entity_stats import _norm_id, strict_prior_by_date
from src.pipeline.features.track_bias_pedigree import (
    _prior_win_rate_by_id_surface,
    _time_sort_seconds,
    build_track_bias_pedigree_features,
    surface_group,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLES = Path("data/page_reference/tables")

CATEGORICAL = ["surface", "direction", "grade", "race_class", "weather", "track_condition", "venue_code", "weight_rule", "course_type",
               "sex", "track_bias_cond_bucket", "track_bias_dist_band"]
RACE_COLS = ["surface", "direction", "grade", "race_class", "weather", "track_condition", "venue_code", "weight_rule", "course_type",
             "distance", "field_size", "round", "horse_number", "bracket_number", "jockey_weight", "weight", "weight_change"]
PRIOR_COLS = [f"{p}_prior_{s}" for p in ("jockey", "trainer") for s in ("races", "wins", "top3", "win_rate", "top3_rate")] + \
             [f"horse_prior_{s}" for s in ("races", "wins", "top3", "win_rate", "top3_rate", "avg_finish")]
BIAS_COLS = ["track_bias_winner_3f_same_day_prior", "track_bias_winner_3f_prev_day", "track_bias_winner_3f_diff", "track_bias_winner_3f_weighted",
             "track_bias_cond_bucket", "track_bias_dist_band", "sire_prior_win_rate_surface", "dam_sire_prior_win_rate_surface",
             "sire_x_track_bias_diff", "dam_sire_x_track_bias_diff", "sire_x_track_bias_weighted", "dam_sire_x_track_bias_weighted"]
DERIVED_COLS = ["sex", "age", "start_hour"]
OUTPUT_COLS = RACE_COLS + PRIOR_COLS + BIAS_COLS + DERIVED_COLS

_HISTORY_COLS = ["race_id", "horse_number", "horse_id", "jockey_id", "trainer_id", "date", "venue_code", "surface", "start_time",
                 "finish_position", "last_3f", "track_condition", "distance"]


def load_history(tables_dir: str | Path | None = None, years: list[int] | None = None) -> pd.DataFrame:
    """race_result_flat（確定結果）を年ごとに読んで連結する。ランナーの起動時に 1 回だけ呼ぶ。"""
    import polars as pl

    base = Path(tables_dir) if tables_dir else DEFAULT_TABLES
    ys = years or sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit())
    parts = []
    for y in ys:
        f = base / str(y) / "race_result_flat.parquet"
        if f.exists():
            sch = pl.read_parquet_schema(f)
            parts.append(pl.read_parquet(f, columns=[c for c in _HISTORY_COLS if c in sch]))
    h = pl.concat(parts, how="diagonal_relaxed").to_pandas()
    h["race_id"] = h["race_id"].astype(str)
    h["horse_number"] = pd.to_numeric(h["horse_number"], errors="coerce")
    sb = base / str(ys[-1]) / "race_shutuba_flat.parquet"
    if sb.exists():
        ped = pl.read_parquet(sb, columns=["sire", "dam_sire"])
        if int((ped["sire"].fill_null("") != "").sum()) or int((ped["dam_sire"].fill_null("") != "").sum()):
            logger.warning("出馬表の sire/dam_sire に値が入っている。学習表は空として作られているので、学習表の作り直しと合わせて serving_layer_a を直すこと")
    return h


def same_day_results(storage: Any, race_ids: list[str]) -> pd.DataFrame:
    """その日すでに確定したレースの結果（``race_result``）を、``load_history`` と同じ列の行にする。履歴の表への追加分（日次エクスポート前）に使う。"""
    from src.scraper.export_tables import _data_to_flat_rows

    rows: list[dict] = []
    for rid in race_ids:
        data = storage.load("race_result", rid, bypass_cache=True)
        if data:
            rows.extend(_data_to_flat_rows({**data, "race_id": data.get("race_id") or rid}, "race_result"))
    if not rows:
        return pd.DataFrame(columns=_HISTORY_COLS)
    df = pd.DataFrame(rows).reindex(columns=_HISTORY_COLS)
    df["race_id"] = df["race_id"].astype(str)
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce")
    return df[df["horse_number"].notna()].reset_index(drop=True)


def _to_pandas(df: Any) -> pd.DataFrame:
    return df.to_pandas() if hasattr(df, "to_pandas") else df.copy()


def _prior_stats(hist: pd.DataFrame, card: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師・馬の通算成績。履歴（結果あり）の行だけを数え、対象行は数えない。"""
    out = card[["race_id", "horse_number"]].copy()
    for id_col, prefix in (("jockey_id", "jockey"), ("trainer_id", "trainer"), ("horse_id", "horse")):
        h = hist[hist[id_col].isin(card[id_col])][["race_id", "horse_number", id_col, "date", "finish_position"]]
        w = pd.concat([h.assign(_t=False), card[["race_id", "horse_number", id_col, "date"]].assign(finish_position=np.nan, _t=True)], ignore_index=True)
        w["date"] = pd.to_datetime(w["date"], errors="coerce")
        w["_gid"] = _norm_id(w[id_col])
        order = ["_gid", "date", "race_id", "horse_number"] if prefix != "horse" else ["_gid", "date", "race_id"]
        w = w.sort_values(order, kind="mergesort")
        fp = pd.to_numeric(w["finish_position"], errors="coerce")
        own = (~w["_t"]).astype(np.float64)
        w["_n"] = own
        w["_win"] = ((fp == 1) & ~w["_t"]).astype(np.float64)
        w["_t3"] = ((fp >= 1) & (fp <= 3) & ~w["_t"]).astype(np.float64)
        w["_fs"] = fp.fillna(0.0) * own
        cs = strict_prior_by_date(w, "_gid", ["_n", "_win", "_t3", "_fs"])
        t = w[w["_t"]]
        r = pd.DataFrame({"race_id": t["race_id"], "horse_number": t["horse_number"]})
        pr = cs.loc[t.index, "_n"]
        pw, pt = cs.loc[t.index, "_win"], cs.loc[t.index, "_t3"]
        prf = pr.replace(0, np.nan)
        if prefix != "horse":
            r[f"{prefix}_prior_races"], r[f"{prefix}_prior_wins"], r[f"{prefix}_prior_top3"] = pr, pw, pt
            r[f"{prefix}_prior_win_rate"], r[f"{prefix}_prior_top3_rate"] = pw / prf, pt / prf
        else:
            r["horse_prior_races"], r["horse_prior_wins"], r["horse_prior_top3"] = pr, pw, pt
            r["horse_prior_win_rate"], r["horse_prior_top3_rate"] = pw / prf, pt / prf
            r["horse_prior_avg_finish"] = np.where(pr > 0, cs.loc[t.index, "_fs"] / prf, np.nan)
        out = out.merge(r.drop_duplicates(["race_id", "horse_number"], keep="last"), on=["race_id", "horse_number"], how="left")
    return out


_GOING_FULL = {"稍": "稍重", "不": "不良"}
_BIAS_ONLY = ["track_bias_winner_3f_same_day_prior", "track_bias_winner_3f_prev_day", "track_bias_winner_3f_diff", "track_bias_winner_3f_weighted",
              "track_bias_cond_bucket", "track_bias_dist_band"]


def _track_bias(hist: pd.DataFrame, card: pd.DataFrame) -> pd.DataFrame:
    """馬場傾向（当日直前・前日）と父/母父の面別勝率。

    馬場傾向は対象日とその前日の行だけで決まるので、その範囲の履歴だけで ``build_track_bias_pedigree_features`` を呼ぶ（全履歴だと 1 分近くかかる）。
    父/母父の面別勝率は全履歴の累計が要るので、同じ関数の部品（``_prior_win_rate_by_id_surface``）を全履歴に直接当てる。"""
    cols = ["race_id", "horse_number", "date", "venue_code", "surface", "start_time", "finish_position", "last_3f", "track_condition", "distance"]
    vc = pd.to_numeric(hist["venue_code"], errors="coerce")
    hb = hist.loc[vc.between(1, 10)].reindex(columns=cols).assign(_pending=False)
    cb = card.reindex(columns=cols).assign(finish_position=np.nan, last_3f=np.nan, _pending=True)
    cb["track_condition"] = cb["track_condition"].replace(_GOING_FULL)       # 出馬表は「稍」「不」。結果表（履歴）と同じ表記にそろえる

    lo = (pd.to_datetime(card["date"]).min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    near = pd.concat([hb[hb["date"].astype(str) >= lo], cb], ignore_index=True)
    sb = near[["race_id", "horse_number"]].assign(sire="", dam_sire="")
    feat = build_track_bias_pedigree_features(near, sb, placeholder_unfinished_races=True, placeholder_win_3f=float("nan"))
    if feat.empty:
        feat = pd.DataFrame(columns=["race_id", "horse_number"] + _BIAS_ONLY)
    out = card[["race_id", "horse_number"]].merge(feat[["race_id", "horse_number"] + _BIAS_ONLY], on=["race_id", "horse_number"], how="left")

    rr = pd.concat([hb, cb], ignore_index=True)
    rr["surface_group"] = rr["surface"].map(surface_group)
    rr["_d"] = pd.to_datetime(rr["date"], errors="coerce").dt.normalize()
    st = rr["start_time"]
    secs = {v: _time_sort_seconds(v) for v in st.unique()}
    rr["_st_sec"] = st.map(secs)
    rr["sire"] = ""
    rr["dam_sire"] = ""
    for role in ("sire", "dam_sire"):
        f = _prior_win_rate_by_id_surface(rr, role, role)
        out = out.merge(f, on=["race_id", "horse_number"], how="left")
    diff, wtd = out["track_bias_winner_3f_diff"], out["track_bias_winner_3f_weighted"]
    for role in ("sire", "dam_sire"):
        rate = out[f"{role}_prior_win_rate_surface"]
        out[f"{role}_x_track_bias_diff"] = rate * diff
        out[f"{role}_x_track_bias_weighted"] = rate * wtd
    return out


def build_layer_a_features(card: Any, history: pd.DataFrame) -> pd.DataFrame:
    """card: 出馬表の行（race_id・horse_id・horse_number・jockey_id・trainer_id・date・venue_code・surface・start_time・track_condition・distance・
    sex_age と OUTPUT_COLS のレース条件・体重など。馬体重は発表後の値）。history: ``load_history`` の結果（対象レースの結果が入っていても除く）。
    戻り値: race_id・horse_id・OUTPUT_COLS（学習表と同じ列名）。カテゴリ列は文字列（欠損は空文字。学習表のカテゴリ水準は空文字を持ち、「(なし)」は持たない）。"""
    c = _to_pandas(card)
    c["race_id"] = c["race_id"].astype(str)
    c["horse_number"] = pd.to_numeric(c["horse_number"], errors="coerce")
    c = c[c["horse_id"].notna() & (c["horse_number"] > 0)].drop_duplicates(["race_id", "horse_number"]).reset_index(drop=True)
    c["horse_number"] = c["horse_number"].astype("int64")
    c["date"] = c["date"].astype(str).str.slice(0, 10)

    h = history[~history["race_id"].astype(str).isin(set(c["race_id"]))]
    h = h[h["date"].astype(str) <= c["date"].max()].copy()
    h["race_id"] = h["race_id"].astype(str)
    h = h[h["horse_number"].notna()].assign(horse_number=lambda d: d["horse_number"].astype("int64"))
    h = h.drop_duplicates(["race_id", "horse_number"], keep="last")          # 日次エクスポート済みの結果と same_day_results が重なっても二重に数えない

    out = c[["race_id", "horse_id"]].copy()
    for col in RACE_COLS:
        out[col] = c[col] if col in c.columns else np.nan
    key = c[["race_id", "horse_number"]]
    out = pd.concat([out, _prior_stats(h, c).drop(columns=["race_id", "horse_number"])], axis=1)
    tb = _track_bias(h, c)
    assert (tb[["race_id", "horse_number"]].to_numpy() == key.to_numpy()).all()
    out = pd.concat([out, tb.drop(columns=["race_id", "horse_number"])], axis=1)

    sex_age = c["sex_age"].astype(str) if "sex_age" in c.columns else pd.Series("", index=c.index)
    out["sex"] = sex_age.str[:1]
    out["age"] = pd.to_numeric(sex_age.str[1:], errors="coerce").astype("float32")
    st = c["start_time"].astype(str) if "start_time" in c.columns else pd.Series("", index=c.index)
    out["start_hour"] = pd.to_numeric(st.str.extract(r"^(\d{1,2})")[0], errors="coerce").astype("float32")
    for col in CATEGORICAL:
        out[col] = out[col].astype("string").fillna("").astype(str)
    return out[["race_id", "horse_id"] + OUTPUT_COLS]
