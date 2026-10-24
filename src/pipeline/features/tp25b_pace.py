"""tp-25b: レース全体のペース予測と、ペース条件付きの馬の能力。

どの特徴量も「そのレースの日より前に分かる情報」だけで作る（ペースの正解・結果はラベルにだけ使い、予測は年ごとの expanding OOF）。
市場オッズ・人気は使わない。

- ``race_targets``: レース単位のペースの正解（前半3F相対・前後半差・RPCI・序盤からの動き）。
- ``course_geo``: コース形状（スタート→最初のカーブ、最後の直線など）を (場, 面, 距離) で引く表。
- ``race_aggregates``: 出走馬の as-of 特徴量（脚質・テン1F・ペース履歴）をレース単位に集約。
- ``pace_slopes``: 馬ごとに「ペースが速い/遅いレースでの成績・上がり」の差（傾き）を縮約付きで as-of に推定。
- ``interaction_features``: 予測ペース × 脚質 × コース形状 の展開相性。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COURSES = ROOT / "data/meta/course_master/jra_courses.json"
FIT_END = pd.Timestamp("2021-12-21")          # horse_history.common.FIT_END と同じ（標準化の基準を学習期間の前半だけで決める）
GOING = {"良": 0, "稍": 1, "重": 2, "不": 3, "稍重": 1, "不良": 3}
START_GEOM = {"main_track": 0, "chute_pocket": 1, "home_stretch_pocket": 2, "straight_course": 3}
TARGETS = ["t_pf_z", "t_bal_z", "t_rpci_z", "t_churn"]
PRED_COLS = ["pp_pf_z", "pp_bal_z", "pp_rpci_z", "pp_churn"]


# ───────────────────────── コース形状
def course_geo(path: str | Path = COURSES) -> pd.DataFrame:
    """(venue_code, surface, distance) → 形状。内回り/外回りは年・出馬表から区別できないので、同じ (場, 面, 距離) の平均を取る。"""
    rows = []
    for c in json.loads(Path(path).read_text(encoding="utf-8")):
        rows.append({"venue_code": str(c["venue_code"]), "surface": c["surface"], "distance": int(c["distance"]),
                     "first_curve_m": c.get("first_curve_m"), "final_straight_m": c.get("final_straight_m"), "lap_length_m": c.get("lap_length_m"),
                     "elevation_range_m": c.get("elevation_range_m"), "start_geom": START_GEOM.get(c.get("start_geometry"), np.nan)})
    d = pd.DataFrame(rows)
    for c in ("first_curve_m", "final_straight_m", "lap_length_m", "elevation_range_m", "start_geom"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["venue_code", "surface", "distance"], as_index=False).agg(
        geo_first_curve=("first_curve_m", "mean"), geo_final_straight=("final_straight_m", "mean"), geo_lap=("lap_length_m", "mean"),
        geo_elev=("elevation_range_m", "mean"), geo_start_geom=("start_geom", "mean"), geo_nvar=("surface", "size"))
    return g


# ───────────────────────── ペースの正解
def race_targets(races: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """races: race_id, date, venue_code, surface, distance, pace_first, pace_second, pace_bal, rpci（1 レース 1 行）。
    labels: stage_labels（race_id, pos_early, pos_late, is_hurdle）。戻り値: race_id + TARGETS。
    t_pf_z: 前半3F を (面, 距離) ごとの平均・標準偏差（FIT_END 以前のレースだけで決める）で標準化。値が小さいほど速い前半。"""
    r = races.drop_duplicates("race_id").copy()
    r["date"] = pd.to_datetime(r["date"])
    fit = r[r["date"] <= FIT_END]
    key = ["surface", "distance"]

    def z(col: str) -> pd.Series:
        st = fit.groupby(key)[col].agg(["mean", "std", "size"]).reset_index()
        st = st[st["size"] >= 20][key + ["mean", "std"]]
        sf = fit.groupby("surface")[col].agg(["mean", "std"]).rename(columns={"mean": "m2", "std": "s2"}).reset_index()   # 距離の組が少ないときは面だけで標準化
        m = r[key].merge(st, on=key, how="left").merge(sf, on="surface", how="left")
        mean = m["mean"].fillna(m["m2"]).to_numpy()
        std = m["std"].fillna(m["s2"]).replace(0, np.nan).to_numpy()
        return pd.Series((r[col].to_numpy(dtype=float) - mean) / std, index=r.index)

    r["t_pf_z"] = z("pace_first")
    r["t_bal_z"] = z("pace_bal")
    r["t_rpci_z"] = z("rpci")
    lab = labels[~labels["is_hurdle"].fillna(False).astype(bool)].copy()
    lab["move"] = (lab["pos_late"] - lab["pos_early"]).abs()
    ch = lab.groupby("race_id")["move"].mean().rename("t_churn").reset_index()
    ch["race_id"] = ch["race_id"].astype(str)
    r["race_id"] = r["race_id"].astype(str)
    out = r.merge(ch, on="race_id", how="left")
    return out[["race_id"] + TARGETS]


# ───────────────────────── レース単位の集約
STYLE_THR = (0.15, 0.25, 0.4)
PACE_HIST = ["pc_pcrel_ewm2", "pc_pci_ewm2", "pc_rpci_ewm3"]


def race_aggregates(df: pd.DataFrame, geo: pd.DataFrame, sx_cols: list[str] | None = None, extra_start: pd.DataFrame | None = None) -> pd.DataFrame:
    """df（馬ごとの行。race_id・条件列・a_style・f2_*・pc_*・sx_* を持つ）→ レース単位の特徴量（race_id をキーに 1 行）。

    extra_start: 別の前段（tp25a）による馬ごとの予測（race_id, horse_id, 列…）。あれば予測由来の集約も足す。"""
    g = df.groupby("race_id", sort=False)
    first = g[["distance", "surface", "venue_code", "track_condition", "grade", "field_size", "date", "direction", "weather"]].first()
    out = pd.DataFrame(index=first.index)
    out["r_distance"] = first["distance"].astype(float)
    out["r_surface"] = first["surface"].map({"芝": 0, "ダート": 1, "障": 2}).astype(float)
    out["r_venue"] = pd.to_numeric(first["venue_code"].astype(str), errors="coerce")
    out["r_going"] = first["track_condition"].astype(str).map(GOING)
    out["r_grade"] = first["grade"].astype(str).map({"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5}).astype(float)
    out["r_field"] = first["field_size"].astype(float)
    d = pd.to_datetime(first["date"])
    out["r_month"] = d.dt.month.astype(float)
    out["r_dir"] = first["direction"].astype(str).map({"右": 0, "左": 1, "直": 2}).astype(float)
    out["r_weather"] = first["weather"].astype(str).map({"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}).astype(float)

    s = df["a_style"].astype(float)
    sg = s.groupby(df["race_id"])
    out["s_known"] = sg.count() / g.size()
    out["s_min"] = sg.min()
    out["s_mean"] = sg.mean()
    out["s_sd"] = sg.std()
    out["s_min2"] = sg.apply(lambda x: np.sort(x.dropna().to_numpy())[1] if x.notna().sum() >= 2 else np.nan)
    out["s_min3m"] = sg.apply(lambda x: np.sort(x.dropna().to_numpy())[:3].mean() if x.notna().sum() >= 3 else np.nan)
    for t in STYLE_THR:
        out[f"s_n_lt{int(t * 100)}"] = (s < t).astype(float).where(s.notna()).groupby(df["race_id"]).sum(min_count=1)
    for c in ("f2_pressure", "f2_nfront", "f2_nfront40", "f2_p_fast"):
        out["r_" + c] = g[c].first()
    for c in PACE_HIST:
        x = df[c].astype(float).groupby(df["race_id"])
        out[f"h_{c}_mean"] = x.mean()
        out[f"h_{c}_max"] = x.max()
        out[f"h_{c}_min"] = x.min()
    if "a_style_late" in df.columns:
        out["s_late_mean"] = df["a_style_late"].astype(float).groupby(df["race_id"]).mean()
    for c in sx_cols or []:
        x = df[c].astype(float).groupby(df["race_id"])
        out["x_" + c] = x.first() if c.startswith("sx_race_") else x.mean()
    if "sx_ten_mean3" in df.columns:
        t = df["sx_ten_mean3"].astype(float).fillna(df["sx_ten_mean5"].astype(float)).fillna(df["sx_ten_last"].astype(float))
        tg = t.groupby(df["race_id"])
        out["x_ten_min"] = tg.min()
        out["x_ten_med"] = tg.median()
        out["x_ten_top3"] = tg.apply(lambda x: np.sort(x.dropna().to_numpy())[:3].mean() if x.notna().sum() >= 3 else np.nan)
        out["x_ten_gap12"] = tg.apply(lambda x: (lambda a: a[1] - a[0])(np.sort(x.dropna().to_numpy())) if x.notna().sum() >= 2 else np.nan)
    if "sx_tenz" in df.columns:
        z = df["sx_tenz"].astype(float)
        zg = z.groupby(df["race_id"])
        out["xz_min"] = zg.min()
        out["xz_sd"] = zg.std()
        out["xz_top3"] = zg.apply(lambda x: np.sort(x.dropna().to_numpy())[:3].mean() if x.notna().sum() >= 3 else np.nan)
        out["xz_n_lt_m10"] = (z < -1.0).astype(float).where(z.notna()).groupby(df["race_id"]).sum(min_count=1)
        out["xz_n_lt_m05"] = (z < -0.5).astype(float).where(z.notna()).groupby(df["race_id"]).sum(min_count=1)
        out["xz_known"] = zg.count() / g.size()
    if extra_start is not None and len(extra_start):
        e = df[["race_id", "horse_id"]].merge(extra_start, on=["race_id", "horse_id"], how="left")
        for c in [c for c in e.columns if c not in ("race_id", "horse_id", "year")]:
            x = e[c].astype(float).groupby(e["race_id"].to_numpy())
            out[f"e_{c}_min"] = x.min()
            out[f"e_{c}_mean"] = x.mean()
            out[f"e_{c}_sd"] = x.std()
            if c.startswith("ts_p_"):
                out[f"e_{c}_sum"] = x.sum(min_count=1)
    geo_k = first[["venue_code", "surface", "distance"]].copy()
    geo_k["venue_code"] = geo_k["venue_code"].astype(str)
    geo_k["surface"] = geo_k["surface"].astype(str)
    geo_k["distance"] = geo_k["distance"].astype(int)
    gj = geo_k.reset_index().merge(geo, on=["venue_code", "surface", "distance"], how="left").set_index("race_id")
    for c in [c for c in gj.columns if c.startswith("geo_")]:
        out[c] = gj[c].reindex(out.index)
    out.index.name = "race_id"
    return out.reset_index()


# ───────────────────────── 馬ごとのペース感応度（as-of）
def pace_slopes(df: pd.DataFrame, actual: pd.DataFrame, half_life: float = 6.0, lam: float = 4.0) -> pd.DataFrame:
    """馬ごとに、過去のレースの『ペース x』と『成績 y』の傾きを縮約付きで推定する。

    df: race_id, horse_id, date, finish_position, field_size と l3f_dev（レース内平均との差の上がり。小さい=速い）
    actual: race_id → x の候補（pf_z, bal_z の列。そのレースの実際のペース。結果が出たあとの過去レースにだけ使う）。
    推定: 直近を重くした重み付き回帰の傾き β = (Sxy − SxSy/W) / (Sxx − Sx²/W + lam)。x は標準化済み（平均 0・分散 1 ほど）なので lam は 約 lam 走分の縮約。
    戻り値は df と同じ行順: sl_<y>_<x>（傾き）, sl_n（過去走数）。自分より前の日付の走だけを使う（同日は使わない）。"""
    d = df[["race_id", "horse_id", "date"]].copy()
    d["_i"] = np.arange(len(d))
    d = d.merge(actual, on="race_id", how="left")
    fp = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy()
    n = pd.to_numeric(df["field_size"], errors="coerce").to_numpy()
    res = 0.5 - (fp - 1.0) / np.maximum(n - 1.0, 1.0)
    res = np.where(fp > 0, res, np.nan)
    ys = {"res": res, "l3f": df["l3f_dev"].astype(float).to_numpy()}
    xs = {c: d.sort_values("_i")[c].to_numpy(dtype=float) for c in actual.columns if c != "race_id"}
    hid = df["horse_id"].astype(str).to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy("datetime64[D]").astype(np.int64)
    order = np.lexsort((df["race_id"].astype(str).to_numpy(), dates, hid))
    decay = 0.5 ** (1.0 / half_life)
    out = {f"sl_{yn}_{xn}": np.full(len(df), np.nan) for yn in ys for xn in xs}
    cnt = np.zeros(len(df))
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and hid[order[j]] == hid[order[i]]:
            j += 1
        st = {(yn, xn): np.zeros(5) for yn in ys for xn in xs}       # W, Sx, Sy, Sxy, Sxx
        k = 0
        pend: list[int] = []
        cur = None
        for t in range(i, j):
            r = order[t]
            if cur is not None and dates[r] != cur:
                for rp in pend:
                    for (yn, xn), s in st.items():
                        x, y = xs[xn][rp], ys[yn][rp]
                        s *= decay
                        if np.isfinite(x) and np.isfinite(y):
                            s += np.array([1.0, x, y, x * y, x * x])
                    k += 1
                pend = []
            cur = dates[r]
            cnt[r] = k
            for (yn, xn), s in st.items():
                W = s[0]
                if W > 0:
                    out[f"sl_{yn}_{xn}"][r] = (s[3] - s[1] * s[2] / W) / (s[4] - s[1] ** 2 / W + lam)
            pend.append(r)
        i = j
    res_df = pd.DataFrame(out, index=df.index)
    res_df["sl_n"] = cnt
    return res_df


# ───────────────────────── 展開相性
def interaction_features(df: pd.DataFrame, pred: pd.DataFrame, slopes: pd.DataFrame | None = None) -> pd.DataFrame:
    """df（馬ごと）に、レース単位の予測ペース pred（race_id + PRED_COLS）を当てて、脚質・形状との交互作用を作る。"""
    p = df[["race_id"]].merge(pred, on="race_id", how="left")
    p.index = df.index
    s = df["a_style"].astype(float)
    out = pd.DataFrame(index=df.index)
    for c in PRED_COLS:
        out[c] = p[c].to_numpy()
    out["ix_style_x_pf"] = s * p["pp_pf_z"]
    out["ix_style_x_bal"] = s * p["pp_bal_z"]
    out["ix_style_x_churn"] = s * p["pp_churn"]
    if "a_style_late" in df.columns:
        out["ix_late_x_bal"] = df["a_style_late"].astype(float) * p["pp_bal_z"]
    out["ix_nfront_x_pf"] = df["f2_nfront"].astype(float) * p["pp_pf_z"]
    out["ix_lone_x_pf"] = df["f2_lone_front"].astype(float) * p["pp_pf_z"]
    gs = s.groupby(df["race_id"])
    out["ix_style_rel"] = s - gs.transform("min")
    out["ix_front_rank"] = gs.rank(method="average")
    gate = (pd.to_numeric(df["horse_number"], errors="coerce") - 1) / (pd.to_numeric(df["field_size"], errors="coerce") - 1).clip(lower=1)
    out["ix_gate_pos"] = gate
    fc = df["geo_first_curve"].astype(float) if "geo_first_curve" in df.columns else pd.Series(np.nan, index=df.index)
    out["ix_gate_x_curve"] = (1.0 - gate) * 100.0 / fc.clip(lower=50)
    out["ix_style_x_curve"] = s * 100.0 / fc.clip(lower=50)
    out["ix_front_x_curve_x_pf"] = (1.0 - s) * 100.0 / fc.clip(lower=50) * p["pp_pf_z"]
    if slopes is not None:
        for c in [c for c in slopes.columns if c.startswith("sl_") and c != "sl_n"]:
            xn = c.split("_", 2)[2]
            pp = {"pf_z": p["pp_pf_z"], "bal_z": p["pp_bal_z"]}.get(xn)
            if pp is None:
                continue
            e = slopes[c].astype(float) * pp
            out["ex_" + c[3:]] = e.to_numpy()
            out["ex_" + c[3:] + "_rel"] = (e - e.groupby(df["race_id"]).transform("mean")).to_numpy()
    return out


def add_tenz(full: pd.DataFrame, fit_end: pd.Timestamp = FIT_END) -> pd.Series:
    """馬の直近テン1F（sx_ten_mean3 → mean5 → last）を、(面, 距離) ごとの平均・標準偏差（FIT_END 以前の行だけで決める）で標準化。小さい=速い。"""
    t = full["sx_ten_mean3"].astype(float).fillna(full["sx_ten_mean5"].astype(float)).fillna(full["sx_ten_last"].astype(float))
    k = pd.DataFrame({"surface": full["surface"].astype(str).to_numpy(), "distance": full["distance"].astype(int).to_numpy(), "t": t.to_numpy()})
    fit = pd.to_datetime(full["date"]).to_numpy() <= np.datetime64(fit_end)
    st = k[fit].groupby(["surface", "distance"])["t"].agg(["mean", "std", "count"]).reset_index()
    st = st[st["count"] >= 100]
    sf = k[fit].groupby("surface")["t"].agg(["mean", "std"]).rename(columns={"mean": "m2", "std": "s2"}).reset_index()
    m = k.merge(st, on=["surface", "distance"], how="left").merge(sf, on="surface", how="left")
    mean, std = m["mean"].fillna(m["m2"]), m["std"].fillna(m["s2"]).replace(0, np.nan)
    return pd.Series(((m["t"] - mean) / std).to_numpy(), index=full.index)
