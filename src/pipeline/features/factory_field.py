"""特徴量ファクトリー G3: レース・出走メンバー構成・コース/環境の候補特徴量（すべて as-of。市場オッズ・人気は使わない）。

入力 df は学習表（修正後）と同じ列を持つ RangeIndex の表で、(date, race_id, horse_number) 順に並べておく。
各 ``grp_*`` は df と同じ index の DataFrame を返す。使う情報は
  * 同じレースの出走馬の「レース前に分かる列」（as-of の強さ・年齢・斤量など）の集約
  * 日付が厳密に前のレースの結果（finish_position）、または同じ日で ``round`` が前のレースの結果
  * 固定情報（コースマスタ・レース名・カレンダー）
だけ。自分のレースと後のレースの結果は一切使わない（tests/pipeline/test_factory_field.py が検査）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.pipeline.models.stage_models import asof_group_mean

ROOT = Path(__file__).resolve().parents[3]
COURSE_MASTER = ROOT / "data/meta/course_master/jra_courses.json"

STRENGTH = ["f3_r", "sr_exp", "sr_perf_ewm2", "c_res_ewm2", "b_aff_all", "horse_prior_win_rate", "jockey_prior_win_rate", "trainer_prior_win_rate",
            "a_l3fz_ewm2", "c_cls_ewm2"]
GOING = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}
WEATHER = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col].astype(object).where(df[col].notna()), errors="coerce").astype("float64")


def _age(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["sex_age"].astype(str).str[1:], errors="coerce")


def _race_stats(df: pd.DataFrame, x: pd.Series, p: str, own: bool = True) -> dict[str, pd.Series]:
    """レース内の集約（平均・最大・最小・標準偏差・1位と2位の差・上位3頭の平均）と自馬との相対値。"""
    g = x.groupby(df["race_id"].to_numpy())
    mean, mx, mn, sd = g.transform("mean"), g.transform("max"), g.transform("min"), g.transform("std")
    rk = g.rank(ascending=False, method="min")
    out = {f"{p}_mean": mean, f"{p}_max": mx, f"{p}_min": mn, f"{p}_std": sd}
    out[f"{p}_gap12"] = mx - x.where(rk == 2).groupby(df["race_id"].to_numpy()).transform("max")
    out[f"{p}_top3mean"] = x.where(rk <= 3).groupby(df["race_id"].to_numpy()).transform("mean")
    if own:
        out[f"{p}_own_mean"] = x - mean
        out[f"{p}_own_max"] = x - mx
        out[f"{p}_own_z"] = (x - mean) / sd.replace(0, np.nan)
        out[f"{p}_own_pct"] = g.rank(pct=True, ascending=True)
        out[f"{p}_n_better"] = rk - 1
    return out


def _frame(d: dict[str, pd.Series], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(d, index=index).astype("float32")


def _key(df: pd.DataFrame, *cols: str) -> pd.Series:
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "|" + df[c].astype(str)
    return s


def _asof(df: pd.DataFrame, key: pd.Series, value: pd.Series, name: str, k: float = 30.0) -> pd.DataFrame:
    t = pd.DataFrame({"date": pd.to_datetime(df["date"]).to_numpy(), "k": key.to_numpy(), "v": value.to_numpy()})
    return asof_group_mean(t, "k", "v", k=k, name=name).set_index(df.index)


# ───────────────────────── 1. 出走メンバーの強さの分布
def grp_mem_strength(df: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for c in STRENGTH:
        if c in df.columns:
            out.update(_race_stats(df, _num(df, c), f"fm_{c}"))
    return _frame(out, df.index)


# ───────────────────────── 2. 出走メンバーの属性の構成
def grp_mem_demo(df: pd.DataFrame) -> pd.DataFrame:
    rid = df["race_id"].to_numpy()
    out: dict[str, pd.Series] = {}
    age = _age(df)
    sex = df["sex_age"].astype(str).str[:1]
    jw, w = _num(df, "jockey_weight"), _num(df, "weight")
    cols = {"age": age, "jw": jw, "wt": w, "wchg": _num(df, "weight_change"), "rest": _num(df, "d_rest"), "nprior": _num(df, "d_nprior"),
            "hrace": _num(df, "horse_prior_races"), "clsprev": _num(df, "c_cls_prev"), "clsstep": _num(df, "c_cls_step"), "jwratio": jw / w * 1000.0}
    for p, x in cols.items():
        out.update(_race_stats(df, x, f"fd_{p}"))
    for nm, m in (("female", sex.eq("牝")), ("gelding", sex.eq("セ")), ("male", sex.eq("牡"))):
        out[f"fd_n_{nm}"] = m.astype(float).groupby(rid).transform("sum")
        out[f"fd_share_{nm}"] = m.astype(float).groupby(rid).transform("mean")
        out[f"fd_own_{nm}"] = m.astype(float)
    for a in (2, 3, 4, 5):
        m = (age == a) if a < 5 else (age >= 5)
        out[f"fd_n_age{a}"] = m.astype(float).groupby(rid).transform("sum")
    out["fd_own_age_rank_share"] = (age.groupby(rid).rank(pct=True))
    first = (_num(df, "d_nprior") == 0) | (_num(df, "horse_prior_races") == 0)
    out["fd_n_first"] = first.astype(float).groupby(rid).transform("sum")
    out["fd_share_first"] = first.astype(float).groupby(rid).transform("mean")
    lay = _num(df, "d_rest") >= 90
    out["fd_n_layoff"] = lay.astype(float).groupby(rid).transform("sum")
    out["fd_own_layoff"] = lay.astype(float)
    up = _num(df, "c_cls_step") > 0
    dn = _num(df, "c_cls_step") < 0
    out["fd_n_classup"], out["fd_n_classdown"] = up.astype(float).groupby(rid).transform("sum"), dn.astype(float).groupby(rid).transform("sum")
    jn = df["jockey_name"].astype(str) if "jockey_name" in df.columns else pd.Series("", index=df.index)
    for nm, mark in (("a3", "▲"), ("a2", "△"), ("a1", "☆"), ("a0", "◇")):
        m = jn.str.contains(mark, regex=False)
        out[f"fd_own_{nm}"] = m.astype(float)
        out[f"fd_n_{nm}"] = m.astype(float).groupby(rid).transform("sum")
    out["fd_own_apprentice"] = jn.str.contains("[▲△☆◇]", regex=True).astype(float)
    out["fd_wchg_abs"] = _num(df, "weight_change").abs()
    out["fd_wchg_rel"] = _num(df, "weight_change") / w
    out["fd_n_wchg_big"] = (_num(df, "weight_change").abs() >= 10).astype(float).groupby(rid).transform("sum")
    return _frame(out, df.index)


# ───────────────────────── 3. 騎手・調教師の格の分布
def grp_mem_conn(df: pd.DataFrame) -> pd.DataFrame:
    rid = df["race_id"].to_numpy()
    out: dict[str, pd.Series] = {}
    for p, c in (("jk", "jockey_prior_win_rate"), ("tr", "trainer_prior_win_rate"), ("jk3", "jockey_prior_top3_rate"), ("tr3", "trainer_prior_top3_rate"),
                 ("jkn", "jockey_prior_races"), ("trn", "trainer_prior_races")):
        out.update(_race_stats(df, _num(df, c), f"fc_{p}"))
    tr = df["trainer_id"].astype(str)
    out["fc_trainer_entries"] = tr.groupby([rid, tr.to_numpy()]).transform("size").astype(float)
    out["fc_n_trainers"] = tr.groupby(rid).transform("nunique").astype(float)
    jk = df["jockey_id"].astype(str)
    out["fc_n_jockeys"] = jk.groupby(rid).transform("nunique").astype(float)
    strength = _num(df, "f3_r")
    best = strength.groupby([rid, tr.to_numpy()]).transform("max")
    out["fc_trainer_best_in_stable"] = (strength >= best).astype(float)
    out["fc_trainer_stable_gap"] = strength - best
    out["fc_top_jockey"] = (_num(df, "jockey_prior_win_rate") >= 0.12).astype(float)
    out["fc_n_top_jockey"] = out["fc_top_jockey"].groupby(rid).transform("sum")
    return _frame(out, df.index)


# ───────────────────────── 4. 脚質の構成と枠の相互作用
def grp_mem_style(df: pd.DataFrame) -> pd.DataFrame:
    rid = df["race_id"].to_numpy()
    out: dict[str, pd.Series] = {}
    st = _num(df, "a_style")
    er = _num(df, "f2_early_rank")
    out.update(_race_stats(df, st, "fs_style"))
    out.update(_race_stats(df, er, "fs_early"))
    for th in (0.2, 0.3, 0.4):
        m = (st < th).astype(float).where(st.notna())
        out[f"fs_n_front{int(th * 100)}"] = m.groupby(rid).transform("sum")
    out["fs_n_late"] = (st > 0.7).astype(float).where(st.notna()).groupby(rid).transform("sum")
    arr, ids = st.to_numpy(), pd.factorize(df["race_id"])[0]
    same_style = np.full(len(df), np.nan)
    nearest_front = np.full(len(df), np.nan)
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
    for a, b in zip(starts[:-1], starts[1:]):
        v = arr[a:b]
        for i in range(b - a):
            if np.isnan(v[i]):
                continue
            o = np.delete(v, i)
            o = o[~np.isnan(o)]
            if len(o):
                same_style[a + i] = float((np.abs(o - v[i]) <= 0.1).sum())
                lo = o[o < v[i]]
                nearest_front[a + i] = float(v[i] - lo.max()) if len(lo) else np.nan
    out["fs_same_style"], out["fs_gap_to_front"] = pd.Series(same_style, index=df.index), pd.Series(nearest_front, index=df.index)
    fs, hn, br = _num(df, "field_size"), _num(df, "horse_number"), _num(df, "bracket_number")
    rel = (hn - 1) / (fs - 1).replace(0, np.nan)
    out["fs_rel_post"] = rel
    out["fs_style_x_inner"] = st * (1 - rel)
    out["fs_front_x_inner"] = ((st < 0.3).astype(float) * (1 - rel)).where(st.notna())
    out["fs_late_x_outer"] = ((st > 0.6).astype(float) * rel).where(st.notna())
    out["fs_early_x_post"] = er * rel
    return _frame(out, df.index)


# ───────────────────────── 5. コースの固定情報（コースマスタ）
_NUM_MASTER = ["lap_length_m", "final_straight_m", "elevation_range_m", "first_curve_m", "n_corners_record_mode", "width_m", "goal_slope_rise"]


def _master(dedupe: bool = True) -> pd.DataFrame:
    rows = json.loads(COURSE_MASTER.read_text(encoding="utf-8"))
    m = pd.DataFrame(rows)
    m["venue_code"] = m["venue_code"].astype(str).str.zfill(2)
    m["distance"] = m["distance"].astype(int)
    for c in _NUM_MASTER:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["start_geo"] = m["start_geometry"].astype("category").cat.codes.astype(float)
    m["radius_cls"] = m["corner_radius_class_A"].astype(str).str.slice(0, 2).astype("category").cat.codes.astype(float)
    m["variant_code"] = m["variant"].map({"": 0, "内": 1, "外": 2, "直": 3}).astype(float)
    m["goal_slope"] = m["goal_slope_exists"].map({True: 1.0, False: 0.0})
    m["variant"] = m["variant"].fillna("").astype(str)
    return m.drop_duplicates(["venue_code", "surface", "distance"], keep="first") if dedupe else m


def course_variant_series(df: pd.DataFrame) -> pd.Series:
    """内/外/''（区別なし）。マスタだけで決まらないレースは NaN（course_variant.py）。"""
    from src.pipeline.features.course_variant import add_course_variant

    return add_course_variant(df, parsed_col=None)


_GEO_COLS = ["start_geo", "radius_cls", "variant_code", "goal_slope"] + _NUM_MASTER


def _geo_join(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series]:
    """mode: first=複数候補なら先頭だけ（旧）/ avg=内外の数値の平均 / var=内外を区別（決まらないレースは平均＋フラグ）。"""
    m = _master(dedupe=False)
    key = ["venue_code", "surface", "distance"]
    k = pd.DataFrame({"venue_code": df["venue_code"].astype(str).str.zfill(2).to_numpy(), "surface": df["surface"].astype(str).to_numpy(),
                      "distance": _num(df, "distance").fillna(-1).astype(int).to_numpy()})
    avg = m.groupby(key)[_NUM_MASTER + ["goal_slope"]].mean().reset_index()
    first = m.drop_duplicates(key, keep="first")[key + ["start_geo", "radius_cls", "variant_code"]]
    base = avg.merge(first, on=key, how="left")
    n_var = m.groupby(key).size().rename("n_variants").reset_index()
    base = base.merge(n_var, on=key, how="left")
    j = k.merge(base, on=key, how="left")
    amb = pd.Series(False, index=df.index)
    if mode == "var":
        var = course_variant_series(df).fillna("?").to_numpy()
        k2 = k.assign(variant=var)
        exact = k2.merge(m[key + ["variant"] + _GEO_COLS], on=key + ["variant"], how="left")
        isamb = (var == "?")
        for c in _GEO_COLS:
            vals = exact[c].to_numpy(dtype=float)
            fb = j[c].to_numpy(dtype=float)
            if c in ("start_geo", "radius_cls", "variant_code"):
                fb = np.full(len(fb), np.nan)                      # 決まらないレースのカテゴリ値は欠損
            j[c] = np.where(isamb, fb, vals)
        amb = pd.Series(isamb & (j["n_variants"].to_numpy(dtype=float) > 1), index=df.index)
    elif mode == "avg":
        for c in ("start_geo", "radius_cls", "variant_code"):
            j[c] = np.nan
    j.index = df.index
    return j, amb


def grp_course_geo_var(df: pd.DataFrame) -> pd.DataFrame:
    j, amb = _geo_join(df, "var")
    return _geo_features(df, j, "fg", amb)


def grp_course_geo_avg(df: pd.DataFrame) -> pd.DataFrame:
    j, _ = _geo_join(df, "avg")
    return _geo_features(df, j.drop(columns=["start_geo", "radius_cls", "variant_code"]), "fga", None)


def _geo_features(df: pd.DataFrame, j: pd.DataFrame, p: str, amb: pd.Series | None) -> pd.DataFrame:
    d = _num(df, "distance")
    cols = [c for c in _GEO_COLS if c in j.columns]
    out = {f"{p}_{c}": j[c] for c in cols}
    out[f"{p}_straight_ratio"] = j["final_straight_m"] / j["lap_length_m"]
    out[f"{p}_first_curve_ratio"] = j["first_curve_m"] / d
    out[f"{p}_laps"] = d / j["lap_length_m"]
    out[f"{p}_dist_after_first_curve"] = d - j["first_curve_m"]
    out[f"{p}_corners_per_1000m"] = j["n_corners_record_mode"] / d * 1000.0
    out[f"{p}_straight_per_dist"] = j["final_straight_m"] / d
    rel = (_num(df, "horse_number") - 1) / (_num(df, "field_size") - 1).replace(0, np.nan)
    out[f"{p}_first_curve_x_post"] = j["first_curve_m"] * (_num(df, "bracket_number") / 8.0)
    out[f"{p}_short_run_x_post"] = (j["first_curve_m"] < 350).astype(float).where(j["first_curve_m"].notna()) * (1 - rel)
    if amb is not None:
        out[f"{p}_variant_ambiguous"] = amb.astype(float)
    return _frame(out, df.index)


def grp_course_geo(df: pd.DataFrame) -> pd.DataFrame:
    """旧版（内外の候補が複数あるときはマスタの先頭の行）。内外を区別した grp_course_geo_var と比べるために残す。"""
    m = _master()
    k = pd.DataFrame({"venue_code": df["venue_code"].astype(str).str.zfill(2).to_numpy(), "surface": df["surface"].astype(str).to_numpy(),
                      "distance": _num(df, "distance").fillna(-1).astype(int).to_numpy()})
    j = k.merge(m[["venue_code", "surface", "distance", "start_geo", "radius_cls", "variant_code", "goal_slope"] + _NUM_MASTER], on=["venue_code", "surface", "distance"], how="left")
    j.index = df.index
    d = _num(df, "distance")
    out = {f"fg_{c}": j[c] for c in ["start_geo", "radius_cls", "variant_code", "goal_slope"] + _NUM_MASTER}
    out["fg_straight_ratio"] = j["final_straight_m"] / j["lap_length_m"]
    out["fg_first_curve_ratio"] = j["first_curve_m"] / d
    out["fg_laps"] = d / j["lap_length_m"]
    out["fg_dist_after_first_curve"] = d - j["first_curve_m"]
    out["fg_corners_per_1000m"] = j["n_corners_record_mode"] / d * 1000.0
    out["fg_straight_per_dist"] = j["final_straight_m"] / d
    out["fg_first_curve_x_post"] = j["first_curve_m"] * (_num(df, "bracket_number") / 8.0)
    out["fg_short_run_x_post"] = (j["first_curve_m"] < 350).astype(float).where(j["first_curve_m"].notna()) * (1 - (_num(df, "horse_number") - 1) / (_num(df, "field_size") - 1).replace(0, np.nan))
    return _frame(out, df.index)


# ───────────────────────── 6. コース別の枠・脚質の有利不利（過去のレースだけの縮約勝率）
def _thirds(rel: pd.Series) -> pd.Series:
    return pd.cut(rel, [-0.01, 0.34, 0.67, 1.01], labels=False).astype("float")


def grp_course_draw(df: pd.DataFrame, use_variant: bool = False, p: str = "fdr") -> pd.DataFrame:
    fin = _num(df, "finish_position")
    win = (fin == 1).astype(float).where(fin > 0)
    t3 = (fin <= 3).astype(float).where(fin > 0)
    fs, hn, br = _num(df, "field_size"), _num(df, "horse_number"), _num(df, "bracket_number")
    rel = (hn - 1) / (fs - 1).replace(0, np.nan)
    th = _thirds(rel)
    ck = _key(df, "venue_code", "surface", "distance")
    if use_variant:                                   # 内回り/外回りを区別（マスタで決まらないレースは「?」）
        ck = ck + "|" + course_variant_series(df).fillna("?").astype(str)
    cond = df["track_condition"].astype(str).map(lambda s: "good" if s == "良" else "heavy")
    st = _num(df, "a_style")
    sb = pd.cut(st, [-0.01, 0.25, 0.5, 0.75, 1.01], labels=False).astype("float")
    out: dict[str, pd.Series] = {}
    base = {}
    for vn, v in (("win", win), ("t3", t3)):
        b = _asof(df, ck, v, f"{p}_course_{vn}")
        out.update({c: b[c] for c in b.columns})
        base[vn] = b[f"{p}_course_{vn}_mean"]
        for kn, kk in (("br", br.astype(str)), ("hn", hn.astype(str)), ("third", th.astype(str)), ("cond_third", cond + "|" + th.astype(str)), ("style", sb.astype(str)),
                       ("style_third", sb.astype(str) + "|" + th.astype(str)), ("fs_third", fs.astype(str) + "|" + th.astype(str))):
            r = _asof(df, ck + "|" + kk, v, f"{p}_{kn}_{vn}")
            out[f"{p}_{kn}_{vn}_mean"], out[f"{p}_{kn}_{vn}_n"] = r[f"{p}_{kn}_{vn}_mean"], r[f"{p}_{kn}_{vn}_n"]
            out[f"{p}_{kn}_{vn}_delta"] = r[f"{p}_{kn}_{vn}_mean"] - base[vn]
    return _frame(out, df.index)


def grp_course_draw_var(df: pd.DataFrame) -> pd.DataFrame:
    return grp_course_draw(df, use_variant=True, p="fdv")


# ───────────────────────── 7. 直近の勝ち馬の傾向（前日まで／同日の先のレース）
def _winner_rows(df: pd.DataFrame) -> pd.DataFrame:
    fs, hn, br = _num(df, "field_size"), _num(df, "horse_number"), _num(df, "bracket_number")
    w = pd.DataFrame({"race_id": df["race_id"].to_numpy(), "date": df["date"].astype(str).to_numpy(), "venue": df["venue_code"].astype(str).to_numpy(),
                      "surface": df["surface"].astype(str).to_numpy(), "round": _num(df, "round").to_numpy(),
                      "style": _num(df, "a_style").to_numpy(), "early": _num(df, "f2_early_rank").to_numpy(), "rel": ((hn - 1) / (fs - 1).replace(0, np.nan)).to_numpy(),
                      "br": br.to_numpy(), "strength": _num(df, "f3_r").to_numpy(), "age": _age(df).to_numpy()})
    w = w[(_num(df, "finish_position") == 1).to_numpy()]
    return w


BIAS_VALS = ["style", "early", "rel", "br", "strength", "age"]


def grp_recent_bias(df: pd.DataFrame) -> pd.DataFrame:
    w = _winner_rows(df)
    n_all = pd.DataFrame({"venue": df["venue_code"].astype(str).to_numpy(), "surface": df["surface"].astype(str).to_numpy(), "date": df["date"].astype(str).to_numpy(),
                          "race_id": df["race_id"].to_numpy(), "round": _num(df, "round").to_numpy()})
    races = n_all.drop_duplicates("race_id")
    # 日別の合計と件数（前日まで: 直近 1 開催日 / 3 開催日 / 6 開催日）
    day = (w.groupby(["venue", "surface", "date"])[BIAS_VALS].agg(["sum", "count"]))
    day.columns = [f"{a}_{b}" for a, b in day.columns]
    day = day.reset_index().sort_values(["venue", "surface", "date"])
    allday = races.groupby(["venue", "surface", "date"]).size().rename("nr").reset_index().sort_values(["venue", "surface", "date"])
    day = allday.merge(day, on=["venue", "surface", "date"], how="left").fillna(0)
    out_cols: dict[str, pd.Series] = {}
    key = ["venue", "surface"]
    feats = {}
    for win_n in (1, 3, 6):
        for v in BIAS_VALS:
            s = day.groupby(key)[f"{v}_sum"].transform(lambda x: x.shift(1).rolling(win_n, min_periods=1).sum())
            c = day.groupby(key)[f"{v}_count"].transform(lambda x: x.shift(1).rolling(win_n, min_periods=1).sum())
            feats[f"frb_prev{win_n}_{v}"] = (s / c.replace(0, np.nan))
        feats[f"frb_prev{win_n}_n"] = day.groupby(key)["nr"].transform(lambda x: x.shift(1).rolling(win_n, min_periods=1).sum())
    pf = pd.concat([day[key + ["date"]], pd.DataFrame(feats, index=day.index)], axis=1)
    # 同日の先のレース（round が前）
    rd = races.merge(w.drop_duplicates("race_id")[["race_id"] + BIAS_VALS], on="race_id", how="left").sort_values(["venue", "surface", "date", "round"])
    g = rd.groupby(["venue", "surface", "date"])
    same: dict[str, pd.Series] = {}
    for v in BIAS_VALS:
        val = rd[v].fillna(0.0)
        cnt = rd[v].notna().astype(float)
        gk = [rd["venue"], rd["surface"], rd["date"]]
        cs = val.groupby(gk).cumsum().groupby(gk).shift(1).fillna(0.0)               # 累積和を 1 行ずらす（『累積和 − 自分』は端数に今回の値の痕跡が残る。tp-34）
        cc = cnt.groupby(gk).cumsum().groupby(gk).shift(1).fillna(0.0)
        same[f"frb_same_{v}"] = cs / cc.replace(0, np.nan)
    same["frb_same_n"] = g.cumcount().astype(float)
    sf = pd.concat([rd[["race_id"]], pd.DataFrame(same, index=rd.index)], axis=1)
    j = pd.DataFrame({"venue": n_all["venue"], "surface": n_all["surface"], "date": n_all["date"], "race_id": n_all["race_id"]}).merge(pf, on=key + ["date"], how="left").merge(sf, on="race_id", how="left")
    j.index = df.index
    own = {"style": _num(df, "a_style"), "early": _num(df, "f2_early_rank"), "rel": (_num(df, "horse_number") - 1) / (_num(df, "field_size") - 1).replace(0, np.nan),
           "br": _num(df, "bracket_number"), "strength": _num(df, "f3_r"), "age": _age(df)}
    out = {c: j[c] for c in j.columns if c.startswith("frb_")}
    for src in ("prev1", "prev3", "same"):
        for v in ("style", "early", "rel", "br"):
            out[f"frb_{src}_{v}_match"] = own[v] - j[f"frb_{src}_{v}"]
            out[f"frb_{src}_{v}_absmatch"] = (own[v] - j[f"frb_{src}_{v}"]).abs()
    return _frame(out, df.index)


# ───────────────────────── 8. カレンダー・時刻・馬場の推移
def grp_calendar_env(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    rid = df["race_id"].astype(str)
    out: dict[str, pd.Series] = {}
    ang = 2 * np.pi * (d.dt.dayofyear.to_numpy(float) - 1.0) / 365.25
    out["fe_sin1"], out["fe_cos1"], out["fe_sin2"], out["fe_cos2"] = np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)
    out["fe_sin3"], out["fe_cos3"] = np.sin(3 * ang), np.cos(3 * ang)
    out["fe_month"], out["fe_weekday"], out["fe_dom"], out["fe_week"] = d.dt.month, d.dt.weekday, d.dt.day, d.dt.isocalendar().week.astype(float)
    out["fe_kai"] = pd.to_numeric(rid.str[6:8], errors="coerce")
    out["fe_nichi"] = pd.to_numeric(rid.str[8:10], errors="coerce")
    out["fe_year_frac"] = d.dt.year + (d.dt.dayofyear - 1) / 365.25
    out["fe_is_monday"] = (d.dt.weekday == 0).astype(float)
    out["fe_is_sunday"] = (d.dt.weekday == 6).astype(float)
    t = df["start_time"].astype(str).str.extract(r"^(\d{1,2}):(\d{2})").astype(float)
    mins = t[0] * 60 + t[1]
    out["fe_start_min"] = mins
    out["fe_start_since_first"] = mins - mins.groupby([df["venue_code"].astype(str).to_numpy(), df["date"].astype(str).to_numpy()]).transform("min")
    rnd = _num(df, "round")
    out["fe_round"] = rnd
    out["fe_round_last"] = (rnd == rnd.groupby([df["venue_code"].astype(str).to_numpy(), df["date"].astype(str).to_numpy()]).transform("max")).astype(float)
    out["fe_round_first"] = (rnd == 1).astype(float)
    out["fe_round_main"] = (rnd == 11).astype(float)
    out["fe_round_after_main"] = (rnd == 12).astype(float)
    going = df["track_condition"].astype(str).map(GOING).astype(float)
    wea = df["weather"].astype(str).map(WEATHER).astype(float)
    out["fe_going"], out["fe_weather"] = going, wea
    out["fe_going_x_weather"] = going * 10 + wea
    out["fe_going_x_surface"] = going * 3 + df["surface"].astype(str).map({"芝": 0, "ダート": 1, "障": 2}).astype(float)
    vk = [df["venue_code"].astype(str).to_numpy(), df["date"].astype(str).to_numpy()]
    first_cond = going.groupby(vk).transform("first")
    out["fe_going_vs_first_race"] = going - first_cond
    out["fe_weather_vs_first_race"] = wea - wea.groupby(vk).transform("first")
    # 前日（同じ場・面で直近の開催日）の最終レースの馬場
    t2 = pd.DataFrame({"venue": df["venue_code"].astype(str).to_numpy(), "surface": df["surface"].astype(str).to_numpy(), "date": df["date"].astype(str).to_numpy(), "going": going.to_numpy(), "round": rnd.to_numpy()})
    last = t2.sort_values(["venue", "surface", "date", "round"]).groupby(["venue", "surface", "date"]).tail(1)[["venue", "surface", "date", "going"]].sort_values(["venue", "surface", "date"])
    last["prev_going"] = last.groupby(["venue", "surface"])["going"].shift(1)
    mj = t2[["venue", "surface", "date"]].merge(last[["venue", "surface", "date", "prev_going"]], on=["venue", "surface", "date"], how="left")
    mj.index = df.index
    out["fe_prev_day_going"] = mj["prev_going"]
    out["fe_going_change_vs_prev_day"] = going - mj["prev_going"]
    out["fe_days_to_weekend"] = (5 - d.dt.weekday).abs()
    return _frame({k: (v if isinstance(v, pd.Series) else pd.Series(v, index=df.index)) for k, v in out.items()}, df.index)


# ───────────────────────── 9. レース種別（名前・条件）と同じ名前のレースの過去
_NAME_FLAGS = {"stakes": "ステークス|S$|Ｓ$", "sho": "賞", "hai": "杯", "cup": "カップ|Ｃ|C$", "kinen": "記念", "tokubetsu": "特別", "handi": "ハンデ|ハンディ", "jump": "ジャンプ|障害|J・",
               "derby": "ダービー|オークス|菊花|皐月|桜花", "roman": "[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|[IVX]+$", "fem": "牝", "city": "ＮＨＫ|NHK|毎日|朝日|日経|読売|スポーツ"}
_CLASS_LEVEL = [("未勝利", 0), ("新馬", 0), ("１勝", 1), ("1勝", 1), ("２勝", 2), ("2勝", 2), ("３勝", 3), ("3勝", 3), ("オープン", 4)]


def _norm_race_name(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.replace(r"第?\d+回|\(.*?\)|（.*?）|\d{4}", "", regex=True).str.replace(r"[ 　]", "", regex=True)
    return t.str.replace(r"(ＧⅠ|ＧⅡ|ＧⅢ|G1|G2|G3|Ｊ・)", "", regex=True)


def grp_race_type(df: pd.DataFrame, pace_table: pd.DataFrame | None = None) -> pd.DataFrame:
    name = df["race_name"].astype(str)
    rc = df["race_class"].astype(str)
    out: dict[str, pd.Series] = {}
    out["fr_name_len"] = name.str.len().astype(float)
    for k, pat in _NAME_FLAGS.items():
        out[f"fr_name_{k}"] = name.str.contains(pat, regex=True).astype(float)
    lvl = pd.Series(np.nan, index=df.index)
    for key, v in _CLASS_LEVEL:
        lvl = lvl.where(~(rc.str.contains(key) & lvl.isna()), float(v))
    out["fr_class_level"] = lvl
    for a in ("２歳", "３歳以上", "３歳", "４歳以上"):
        out[f"fr_age_{a}"] = rc.str.contains(a, regex=False).astype(float)
    out["fr_age_limit_code"] = np.select([rc.str.contains("２歳"), rc.str.contains("３歳以上"), rc.str.contains("３歳"), rc.str.contains("４歳以上")], [2, 35, 3, 45], 0).astype(float)
    out["fr_rule_code"] = df["weight_rule"].astype(str).map({"馬齢": 0, "定量": 1, "別定": 2, "ハンデ": 3}).astype(float)
    out["fr_grade_code"] = df["grade"].astype(str).map({"": 0, "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5, "OP": 6}).astype(float)
    d = _num(df, "distance")
    out["fr_dist_k"] = d / 1000.0
    out["fr_dist_log"] = np.log(d)
    out["fr_dist_mod400"] = d % 400
    out["fr_dist_mod200"] = d % 200
    out["fr_is_sprint"], out["fr_is_mile"], out["fr_is_mid"], out["fr_is_long"] = (d <= 1400).astype(float), ((d > 1400) & (d <= 1800)).astype(float), ((d > 1800) & (d <= 2200)).astype(float), (d > 2200).astype(float)
    fs = _num(df, "field_size")
    out["fr_full"], out["fr_small"] = (fs >= 16).astype(float), (fs <= 9).astype(float)
    out["fr_field_bin"] = pd.cut(fs, [0, 7, 9, 12, 15, 17, 30], labels=False).astype(float)
    out["fr_surface_x_dist"] = df["surface"].astype(str).map({"芝": 1, "ダート": 2, "障": 3}).astype(float) * 10000 + d
    out["fr_dir_code"] = df["direction"].astype(str).map({"右": 0, "左": 1, "直": 2, "": 3}).astype(float)
    # 同じ名前のレースの過去（前の開催）の勝ち馬の傾向
    nm = _norm_race_name(name)
    fin = _num(df, "finish_position")
    for p, x in (("age", _age(df)), ("style", _num(df, "a_style")), ("strength", _num(df, "f3_r")), ("weight", _num(df, "weight")), ("fs", fs)):
        v = x.where(fin == 1)
        r = _asof(df, nm, v, f"frn_win_{p}", k=1.0)
        out[f"frn_win_{p}_mean"], out[f"frn_win_{p}_n"] = r[f"frn_win_{p}_mean"], r[f"frn_win_{p}_n"]
        if p in ("age", "style", "strength", "weight"):
            out[f"frn_own_vs_win_{p}"] = x - r[f"frn_win_{p}_mean"]
    # コース × クラスのペース・上がりの基準（過去のレースだけ）
    if pace_table is not None:
        out.update(_pace_baseline(df, pace_table))
    return _frame(out, df.index)


def _pace_baseline(df: pd.DataFrame, pace_table: pd.DataFrame) -> dict[str, pd.Series]:
    pt = pace_table.drop_duplicates("race_id").set_index("race_id")
    j = pt.reindex(df["race_id"].to_numpy())
    j.index = df.index
    rc = df["race_class"].astype(str).str.replace(r"サラ系|[２３４]歳(以上)?", "", regex=True).str.strip()
    out: dict[str, pd.Series] = {}
    ck = _key(df, "venue_code", "surface", "distance")
    first = df.groupby("race_id", sort=False).cumcount() == 0                  # レース単位で数えるため先頭馬の行だけに値を置く
    for c in [x for x in ("pace_first", "pace_second", "rpci", "win_l3f") if x in j.columns]:
        v = j[c].where(first)
        for nm, key in (("course", ck), ("course_cls", ck + "|" + rc)):
            r = _asof(df, key, v, f"frp_{nm}_{c}", k=10.0)
            out[f"frp_{nm}_{c}_mean"], out[f"frp_{nm}_{c}_n"] = r[f"frp_{nm}_{c}_mean"], r[f"frp_{nm}_{c}_n"]
        out[f"frp_style_x_{c}"] = (_num(df, "a_style") - 0.5) * out[f"frp_course_{c}_mean"]       # 全期間の平均で中心化しない（未来が混ざる）
    return out


# ───────────────────────── 10. 単純な組合せ・比率・偶奇など
def grp_cross_misc(df: pd.DataFrame) -> pd.DataFrame:
    fs, hn, br = _num(df, "field_size"), _num(df, "horse_number"), _num(df, "bracket_number")
    d, w, jw = _num(df, "distance"), _num(df, "weight"), _num(df, "jockey_weight")
    age = _age(df)
    rid = df["race_id"].to_numpy()
    out: dict[str, pd.Series] = {}
    out["fx_post_from_outside"] = fs - hn
    out["fx_bracket_rel"] = br / 8.0
    out["fx_hn_odd"] = (hn % 2).astype(float)
    out["fx_hn_first"], out["fx_hn_last"] = (hn == 1).astype(float), (hn == fs).astype(float)
    out["fx_inner3"], out["fx_outer3"] = (hn <= 3).astype(float), (fs - hn < 3).astype(float)
    out["fx_bracket_size"] = br.groupby([rid, br.to_numpy()]).transform("size").astype(float)
    out["fx_age_x_dist"] = age * d / 1000.0
    out["fx_w_per_dist"] = w / d
    out["fx_jw_vs_median"] = jw - jw.groupby(rid).transform("median")
    out["fx_jw_rank"] = jw.groupby(rid).rank(ascending=False, method="average")
    out["fx_w_vs_median"] = w - w.groupby(rid).transform("median")
    out["fx_w_rank"] = w.groupby(rid).rank(ascending=False, method="average")
    out["fx_jw_x_dist"] = jw * d / 1000.0
    out["fx_jw_minus_age_std"] = jw - df["sex_age"].astype(str).str[:1].map({"牡": 57.0, "牝": 55.0, "セ": 57.0}).astype(float)
    out["fx_wchg_sign"] = np.sign(_num(df, "weight_change"))
    out["fx_wchg_x_dist"] = _num(df, "weight_change") * d / 1000.0
    out["fx_w_bin10"] = (w // 10).astype(float)
    out["fx_surface_dir"] = df["surface"].astype(str).map({"芝": 0, "ダート": 1, "障": 2}).astype(float) * 10 + df["direction"].astype(str).map({"右": 0, "左": 1, "直": 2, "": 3}).astype(float)
    out["fx_strength_x_field"] = _num(df, "f3_r") * fs
    out["fx_exp_x_age"] = _num(df, "horse_prior_races") / age
    out["fx_races_per_age"] = _num(df, "horse_prior_races") / (age - 1).clip(lower=1)
    return _frame(out, df.index)


GROUPS: dict[str, Callable[..., pd.DataFrame]] = {
    "mem_strength": grp_mem_strength,
    "mem_demo": grp_mem_demo,
    "mem_conn": grp_mem_conn,
    "mem_style": grp_mem_style,
    "course_geo": grp_course_geo,
    "course_geo_var": grp_course_geo_var,
    "course_geo_avg": grp_course_geo_avg,
    "course_draw": grp_course_draw,
    "course_draw_var": grp_course_draw_var,
    "recent_bias": grp_recent_bias,
    "calendar_env": grp_calendar_env,
    "race_type": grp_race_type,
    "cross_misc": grp_cross_misc,
}


def build_group(name: str, df: pd.DataFrame, **kw) -> pd.DataFrame:
    fn = GROUPS[name]
    return fn(df, **kw) if name == "race_type" else fn(df)


def load_pace_table(tables_dir: str | Path = ROOT / "data/page_reference/tables", years=range(2020, 2026)) -> pd.DataFrame:
    """race_result_race の pace(JSON)・lap_times・勝ち馬の上がりから、レース単位のペース表を作る。"""
    import polars as pl

    parts = []
    for y in years:
        f = Path(tables_dir) / str(y) / "race_result_race.parquet"
        if f.exists():
            parts.append(pl.read_parquet(f, columns=["race_id", "pace", "lap_times"]))
    t = pl.concat(parts, how="diagonal_relaxed")
    t = t.with_columns(pl.col("pace").str.json_path_match("$.first_half_3f").cast(pl.Float64, strict=False).alias("pace_first"),
                       pl.col("pace").str.json_path_match("$.second_half_3f").cast(pl.Float64, strict=False).alias("pace_second"))
    t = t.with_columns((pl.col("pace_first") - pl.col("pace_second")).alias("rpci"))
    return t.select("race_id", "pace_first", "pace_second", "rpci").to_pandas()
