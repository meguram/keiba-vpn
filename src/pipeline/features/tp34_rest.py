"""tp-34: 休養明け・レース間隔の特徴量（すべて as-of。そのレース日より前の走・調教だけから作る）。

G1（factory_self.g_rest / g_experience）に既にある列（前走間隔の日数・対数・フラグ、直近 N 日の出走数、休養明けからの走数 rest_since_layoff など）とは重ならない、
『間隔の帯』『叩き何戦目』『馬ごと・厩舎ごとの休み明けの得意不得意』『他の属性との掛け合わせ』『前走の消耗』『休養中の調教量』『季節』を作る。

群:
  R1_band        間隔の帯（連闘〜1 週 / 2 週 / 3〜4 週 / 5〜8 週 / 9〜13 週 / 14〜26 週 / 半年超）、叩き何戦目、その馬自身の過去の間隔との比較
  R2_horse_pref  馬ごとの『同じ間隔の帯 / 休み明け / 叩き何戦目』での過去成績（馬の全体平均へ縮約した得意度）
  R3_tataki_gain 馬ごとの『叩き 2 戦目の上昇度』『3 戦目の上昇度』（休み明け初戦と叩き後の成績差の過去平均）
  R4_trainer     調教師ごとの『休み明け初戦 / 叩き 2 戦目 / 間隔の帯』の成績（調教師の全体平均との差）
  R5_cross       年齢・クラス・面・季節 × 間隔の帯 の全馬横断の as-of 成績（属性だけの成績との差）
  R6_prev        前走の消耗（着差・上がり・押し上げ・時計）と間隔の組み合わせ
  R7_work        前走から今回までの調教の本数・強度・カバー率（horse_training 由来。記録の無い馬は欠損）
  R8_season      前走月 × 今回月（夏休み明け・年末年始）と、その as-of 成績
キーは race_id / horse_id（調教師は trainer_id）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs
from src.pipeline.models import stage_models as sm

KEY = ["race_id", "horse_id"]
BAND_EDGES = [7, 14, 28, 56, 91, 182]
BAND_NAMES = ["連闘〜1週", "2週", "3〜4週", "5〜8週", "9〜13週", "14〜26週", "半年超"]
LAYOFF_DAYS = 90


def band_id(gap: np.ndarray) -> np.ndarray:
    g = np.asarray(gap, dtype=float)
    b = np.digitize(np.where(np.isfinite(g), g, 0.0), BAND_EDGES, right=True).astype(float)
    return np.where(np.isfinite(g), b, np.nan)


def prepare(runs: pd.DataFrame) -> pd.DataFrame:
    """runs（factory_self.load_runs の形）に、間隔の帯・叩き何戦目などの『今回のレース前に分かる』派生列を足す。"""
    d = fs.prep(runs)
    gap = (d["dnum"] - fs.lag(d, "dnum")).astype(float)
    d["gap"] = gap
    d["band"] = band_id(gap.to_numpy())
    new = (gap >= LAYOFF_DAYS) | gap.isna()
    spell = new.groupby(d["horse_id"], sort=False).cumsum()
    tat = d.groupby([d["horse_id"], spell.to_numpy()], sort=False).cumcount() + 1
    d["tataki"] = np.where(gap.isna(), np.nan, tat.to_numpy(float))                  # 馬のデータ上の初走（デビューか不明）は欠損
    d["is_lay"] = np.where(gap.notna(), (gap >= LAYOFF_DAYS).astype(float), np.nan)
    d["is_lay60"] = np.where(gap.notna(), (gap >= 60).astype(float), np.nan)
    d["tat_cat"] = pd.cut(d["tataki"], [0, 1, 2, 3, 99], labels=["T1", "T2", "T3", "T4p"]).astype("string").fillna("T0")
    d["band_s"] = pd.Series(d["band"]).map(lambda x: "B?" if not np.isfinite(x) else f"B{int(x)}")
    d["age_c"] = d["age"].clip(upper=7).fillna(-1).astype(int).astype(str)
    d["cls_s"] = d["cls"].fillna(-1).astype(int).astype(str)
    d["surf_s"] = d["surface"].astype(str)
    d["month_s"] = d["month"].astype(int).astype(str)
    return d


# ───────────────────────── R1
def _past_gap_stats(d: pd.DataFrame) -> pd.DataFrame:
    """その馬の過去の間隔の平均・中央値・今回の間隔の順位（過去の間隔のうち今回より短かった割合）。

    過去の走それぞれの『直前の間隔』は d["gap"] の start..i-1 行。i 行目自身の gap は今回の間隔なので過去側には含めない。"""
    h = d["horse_id"].to_numpy()
    gap = d["gap"].to_numpy(float)
    n = len(d)
    mean_p = np.full(n, np.nan)
    med_p = np.full(n, np.nan)
    pct = np.full(n, np.nan)
    cnt = np.zeros(n)
    start = 0
    for i in range(n):
        if i == 0 or h[i] != h[i - 1]:
            start = i
        arr = gap[start:i]
        arr = arr[np.isfinite(arr)]
        cnt[i] = len(arr)
        if len(arr):
            mean_p[i] = arr.mean()
            med_p[i] = np.median(arr)
            if np.isfinite(gap[i]):
                pct[i] = float((arr < gap[i]).mean() + 0.5 * (arr == gap[i]).mean())
    return pd.DataFrame({"mean": mean_p, "med": med_p, "pct": pct, "n": cnt}, index=d.index)


def g_band(d: pd.DataFrame, **_) -> pd.DataFrame:
    st = _past_gap_stats(d)
    gap = d["gap"]
    out = fs._f(d, rs1_band=d["band"], rs1_tataki=d["tataki"], rs1_tataki_cap4=d["tataki"].clip(upper=4), rs1_is_lay60=d["is_lay60"], rs1_is_lay=d["is_lay"],
                rs1_second_run=(d["tataki"] == 2).astype(float).where(d["tataki"].notna()), rs1_third_run=(d["tataki"] == 3).astype(float).where(d["tataki"].notna()),
                rs1_gap_vs_mean=gap - st["mean"], rs1_gap_ratio=gap / st["mean"].where(st["mean"] > 0), rs1_gap_vs_med=gap - st["med"], rs1_gap_pct=st["pct"], rs1_n_past_gaps=st["n"])
    out["rs1_gap_z"] = (gap - st["mean"]) / np.maximum(st["mean"], 7.0)
    return out


# ───────────────────────── R2
def g_horse_pref(d: pd.DataFrame, **_) -> pd.DataFrame:
    parts = [fs.seg_affinity(d, d["band_s"], "rs2_band", vals=("fp", "win", "top3")),
             fs.seg_affinity(d, d["is_lay"].map(lambda x: "L" if x == 1 else ("N" if x == 0 else "?")), "rs2_lay", vals=("fp", "top3")),
             fs.seg_affinity(d, d["tat_cat"], "rs2_tat", vals=("fp", "top3"))]
    return pd.concat(parts, axis=1)


# ───────────────────────── R3
def g_tataki_gain(d: pd.DataFrame, **_) -> pd.DataFrame:
    """叩き 2 戦目の fp が休み明け初戦（1 走前）よりどれだけ良くなったか（fp は小さいほど良いので、前 − 今 を『上昇』とする）の、過去の平均。"""
    fp, fp1 = d["fp"].to_numpy(float), fs.lag(d, "fp").to_numpy(float)
    fp2 = fs.lag(d, "fp", 2).to_numpy(float)
    t = d["tataki"].to_numpy(float)
    gain2 = np.where(t == 2, fp1 - fp, np.nan)
    gain3 = np.where(t == 3, fp1 - fp, np.nan)
    gain3b = np.where(t == 3, fp2 - fp, np.nan)                       # 休み明け初戦からの上昇
    tmp = d.assign(_g2=gain2, _g3=gain3, _g3b=gain3b)
    out = {}
    k = 3.0
    for nm, col in (("rs3_gain2", "_g2"), ("rs3_gain3", "_g3"), ("rs3_gain3_from1", "_g3b")):
        s = fs.excl_cum(tmp, ["horse_id"], col)
        n = fs.excl_cum(tmp, ["horse_id"], col, as_count=True)
        out[f"{nm}_mean"] = s / (n + k)
        out[f"{nm}_n"] = n
    r = fs._f(d, **out)
    r["rs3_gain2_if_second"] = r["rs3_gain2_mean"].where(d["tataki"] == 2)
    r["rs3_gain3_if_third"] = r["rs3_gain3_mean"].where(d["tataki"] == 3)
    return r


# ───────────────────────── R4 / R5 / R8: 全馬横断の as-of 成績
def _grp(d: pd.DataFrame, key: pd.Series, value: str, name: str, k: float = 30.0) -> pd.DataFrame:
    tmp = pd.DataFrame({"date": d["date"].to_numpy(), "g": key.astype(str).to_numpy(), "v": d[value].to_numpy(float)})
    return sm.asof_group_mean(tmp, "g", "v", k=k, name=name).set_axis(d.index)             # serving_v4.fast_asof が差し替えられるよう、モジュール経由で呼ぶ


def _lift(d: pd.DataFrame, cross_key: pd.Series, base_key: pd.Series, value: str, name: str, k: float = 30.0) -> pd.DataFrame:
    c = _grp(d, cross_key, value, "c", k)
    b = _grp(d, base_key, value, "b", k)
    return pd.DataFrame({f"{name}_{value}": c["c_mean"] - b["b_mean"], f"{name}_n": c["c_n"]}, index=d.index)


def g_trainer(d: pd.DataFrame, **_) -> pd.DataFrame:
    tr = d["trainer_id"].astype(str)
    parts = []
    for v in ("fp", "top3"):
        parts.append(_lift(d, tr + "|" + d["tat_cat"], tr, v, "rs4_tr_tat", k=20.0))
        parts.append(_lift(d, tr + "|" + d["band_s"], tr, v, "rs4_tr_band", k=20.0))
    parts.append(_lift(d, tr + "|" + d["is_lay"].map(lambda x: "L" if x == 1 else ("N" if x == 0 else "?")), tr, "fp", "rs4_tr_lay", k=20.0))
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def g_cross(d: pd.DataFrame, **_) -> pd.DataFrame:
    b = d["band_s"]
    spec = {"rs5_age_band": (d["age_c"] + "|" + b, d["age_c"]), "rs5_cls_band": (d["cls_s"] + "|" + b, d["cls_s"]), "rs5_surf_band": (d["surf_s"] + "|" + b, d["surf_s"]),
            "rs5_month_lay": (d["month_s"] + "|" + d["is_lay60"].astype(str), d["month_s"]), "rs5_age_tat": (d["age_c"] + "|" + d["tat_cat"], d["age_c"]),
            "rs5_cls_tat": (d["cls_s"] + "|" + d["tat_cat"], d["cls_s"]), "rs5_sex_band": (d["sex"].astype(str) + "|" + b, d["sex"].astype(str)),
            "rs5_dist_band": ((d["dist"].fillna(0) // 400).astype(int).astype(str) + "|" + b, (d["dist"].fillna(0) // 400).astype(int).astype(str))}
    parts = []
    for nm, (ck, bk) in spec.items():
        for v in ("fp", "top3"):
            parts.append(_lift(d, ck, bk, v, nm))
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def g_season(d: pd.DataFrame, **_) -> pd.DataFrame:
    pm = fs.lag(d, "month")
    cm = d["month"].astype(float)
    gap = d["gap"]
    out = fs._f(d, rs8_month=cm, rs8_prev_month=pm, rs8_summer_return=(pm.between(6, 8) & cm.between(9, 11) & (gap >= 40)).astype(float).where(pm.notna()),
                rs8_winter_break=(pm.between(11, 12) & cm.between(1, 3) & (gap >= 21)).astype(float).where(pm.notna()),
                rs8_spring_break=(pm.between(3, 5) & cm.between(6, 8) & (gap >= 56)).astype(float).where(pm.notna()))
    key = pm.fillna(0).astype(int).astype(str) + ">" + d["month"].astype(int).astype(str)
    for v in ("fp", "top3"):
        c = _grp(d, key, v, "c", 40.0)
        out[f"rs8_pm_cm_{v}"] = c["c_mean"].to_numpy()
    out["rs8_pm_cm_n"] = _grp(d, key, "fp", "c", 40.0)["c_n"].to_numpy()
    return out


# ───────────────────────── R6
def g_prev(d: pd.DataFrame, **_) -> pd.DataFrame:
    gap = d["gap"]
    sg = np.sqrt(np.maximum(gap, 1.0))
    pm1, pm2 = fs.lag(d, "margin_len"), fs.lag(d, "margin_len", 2)
    n = fs.excl_cum(d, ["horse_id"], "fp", as_count=True)
    mean_fp = np.where(n > 0, fs.excl_cum(d, ["horse_id"], "fp") / np.maximum(n, 1), np.nan)
    out = fs._f(d, rs6_prev_fp=fs.lag(d, "fp"), rs6_prev_margin=pm1, rs6_prev_l3f_gap=fs.lag(d, "l3f_gap"), rs6_prev_pos_gain=fs.lag(d, "pos_gain"),
                rs6_prev_time_gap=fs.lag(d, "time_gap200"), rs6_margin_per_sqrt_gap=pm1 / sg, rs6_margin_per_week=pm1 / np.maximum(gap, 1.0) * 7.0,
                rs6_prev_hard=(pm1 >= 4).astype(float).where(pm1.notna()), rs6_two_hard=((pm1 >= 4) & (pm2 >= 4)).astype(float).where(pm1.notna() & pm2.notna()),
                rs6_prev_fp_vs_mean=fs.lag(d, "fp") - pd.Series(mean_fp, index=d.index), rs6_prev_win=fs.lag(d, "win"), rs6_prev_top3=fs.lag(d, "top3"))
    out["rs6_prevfp_x_laggap"] = out["rs6_prev_fp"] * np.log1p(gap)
    out["rs6_prev_dist_x_gap"] = fs.lag(d, "dist") / 1000.0 / np.maximum(gap, 1.0) * 7.0
    return out


# ───────────────────────── R7
def work_table(laps: pd.DataFrame) -> pd.DataFrame:
    """追い切り（training_laps）を (horse_id, race_date) ごとに集計する。train_date < race_date の本数・強度・種別。"""
    L = laps[["horse_id", "train_date", "race_date", "kind", "leg_color", "days_before"]].copy()
    L["horse_id"] = L["horse_id"].astype(str)
    L = L[pd.to_datetime(L["train_date"]) < pd.to_datetime(L["race_date"])]
    lc = L["leg_color"].astype(str)
    L["hard"] = lc.str.contains("一杯|強|G強", regex=True).astype(float)
    L["slope"] = (L["kind"] == "slope").astype(float)
    L["wood"] = (L["kind"] == "wood").astype(float)
    L["poly"] = (L["kind"] == "poly").astype(float)
    g = L.groupby(["horse_id", "race_date"], sort=False)
    t = g.agg(n_work=("days_before", "size"), max_db=("days_before", "max"), min_db=("days_before", "min"), n_hard=("hard", "sum"), n_slope=("slope", "sum"),
              n_wood=("wood", "sum"), n_poly=("poly", "sum")).reset_index()
    t["race_date"] = pd.to_datetime(t["race_date"]).dt.strftime("%Y-%m-%d")
    return t


def g_work(d: pd.DataFrame, laps: pd.DataFrame | None = None, work: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    if work is None:
        if laps is None:
            return pd.DataFrame(index=d.index)
        work = work_table(laps)
    key = pd.DataFrame({"horse_id": d["horse_id"].astype(str).to_numpy(), "race_date": d["date"].dt.strftime("%Y-%m-%d").to_numpy()})
    m = key.merge(work, on=["horse_id", "race_date"], how="left")
    m.index = d.index
    gap = d["gap"]
    n = m["n_work"].astype(float)
    mx = m["max_db"].astype(float)
    out = fs._f(d, rs7_n_work=n, rs7_max_days_before=mx, rs7_min_days_before=m["min_db"], rs7_work_per_week=n / np.maximum(mx, 7.0) * 7.0,
                rs7_n_hard=m["n_hard"], rs7_hard_share=m["n_hard"] / n.where(n > 0), rs7_slope_share=m["n_slope"] / n.where(n > 0), rs7_wood_share=m["n_wood"] / n.where(n > 0),
                rs7_poly_share=m["n_poly"] / n.where(n > 0), rs7_n_per_gap_week=n / np.maximum(gap, 7.0) * 7.0, rs7_cover=mx / np.maximum(gap, 1.0))
    tmp = d.assign(_n=n.to_numpy())
    sm, cn = fs.excl_cum(tmp, ["horse_id"], "_n"), fs.excl_cum(tmp, ["horse_id"], "_n", as_count=True)
    out["rs7_nwork_vs_past"] = n.to_numpy() - np.where(cn > 0, sm / np.maximum(cn, 1), np.nan)
    out["rs7_lay_lowwork"] = ((gap >= 60) & (n <= 2)).astype(float).where(n.notna() & gap.notna())
    out["rs7_lay_heavywork"] = ((gap >= 60) & (n >= 8)).astype(float).where(n.notna() & gap.notna())
    return out


GROUPS = {"R1_band": g_band, "R2_horse_pref": g_horse_pref, "R3_tataki_gain": g_tataki_gain, "R4_trainer": g_trainer, "R5_cross": g_cross,
          "R6_prev": g_prev, "R7_work": g_work, "R8_season": g_season}


def build_all(runs: pd.DataFrame, laps: pd.DataFrame | None = None, only: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """runs: load_runs の形（馬・日付順、RangeIndex）。laps: training_laps（R7 用。無ければ R7 は作らない）。戻り値: {群名: 特徴量表（KEY 付き）}。"""
    d = prepare(runs)
    out = {}
    for name, fn in GROUPS.items():
        if only and name not in only:
            continue
        f = fn(d, laps=laps)
        if f.shape[1]:
            out[name] = f.astype("float32").assign(race_id=d["race_id"].to_numpy(), horse_id=d["horse_id"].to_numpy())
    return out
