"""追い切りラップ → (1) 最終・1週前の標準化した時計と強度、(2) 調教パターンと調教師別の採点（tp-19）。

すべて as-of: レース日より前の調教だけ、成績は前日までに確定したレースだけを使う。キーは horse_id / trainer_id。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMED = ("slope", "wood", "poly")
INTENSITY = {"馬也": 0, "馬なり": 0, "強め": 1, "強": 1, "仕掛": 1, "G強": 1, "直強": 1, "G仕": 1, "一杯": 2, "G一": 2, "直一": 2, "一": 2}
RANK = {"A": 4, "B": 3, "C": 2, "D": 1}
REF_COLS = ("t4", "t3", "t1", "accel")


def prepare(laps: pd.DataFrame) -> pd.DataFrame:
    t = laps[laps["days_before"] >= 1].copy()
    t["train_date"] = pd.to_datetime(t["train_date"])
    t["race_date"] = pd.to_datetime(t["race_date"]).dt.strftime("%Y-%m-%d")
    t["intensity"] = t["leg_color"].map(INTENSITY)
    t["rank_n"] = t["rank"].map(RANK)
    t["timed"] = t["kind"].isin(TIMED) & t["t1"].notna()
    return t


GRP = ["kind", "center", "intensity"]


def _ym(t: pd.DataFrame) -> pd.Series:
    return (t["train_date"].dt.year * 12 + t["train_date"].dt.month).astype("int64")


def build_reference(t: pd.DataFrame, window_months: int = 12, min_n: int = 200) -> pd.DataFrame:
    """(種別, 所属, 強度, 月) ごとの、直前 window_months か月（その月は含めない）の平均・標準偏差。列: GRP + ym + {t4,t3,t1,accel}_{mu,sd}。"""
    t = t.copy()
    t["ym"] = _ym(t)
    timed = t[t["timed"] & t["intensity"].notna()]
    ref_all = None
    for c in REF_COLS:
        m = timed.groupby(GRP + ["ym"])[c].agg(["count", "sum", lambda s: float((s**2).sum())])
        m.columns = ["n", "s", "ss"]
        m = m.reset_index()
        res = []
        for key, g in m.groupby(GRP):
            g = g.set_index("ym").sort_index()
            full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 2), fill_value=0)
            roll = full[["n", "s", "ss"]].rolling(window_months, min_periods=1).sum().shift(1).fillna(0)
            mean = roll["s"] / roll["n"].replace(0, np.nan)
            sd = np.sqrt((roll["ss"] / roll["n"].replace(0, np.nan) - mean**2).clip(lower=1e-6))
            ok = roll["n"] >= min_n
            res.append(pd.DataFrame({"kind": key[0], "center": key[1], "intensity": key[2], "ym": full.index, f"{c}_mu": mean.where(ok), f"{c}_sd": sd.where(ok)}))
        ref = pd.concat(res, ignore_index=True)
        ref["ym"] = ref["ym"].astype("int64")
        ref_all = ref if ref_all is None else ref_all.merge(ref, on=GRP + ["ym"], how="outer")
    return ref_all


def apply_reference(t: pd.DataFrame, ref: pd.DataFrame, asof: bool = False) -> pd.DataFrame:
    """参照表で t4/t3/t1/accel を標準化する（小さい = 速い）。asof=True（推論用）は、参照表に無い月を、その種別で直近の月の値で補う。"""
    t = t.copy()
    t["ym"] = _ym(t)
    base = t[GRP + ["ym"]].copy()
    base["_i"] = np.arange(len(base))
    mm = base.merge(ref, on=GRP + ["ym"], how="left").sort_values("_i")
    if asof:
        miss = mm[f"{REF_COLS[0]}_mu"].isna().to_numpy()
        if miss.any():
            r2 = ref.dropna(subset=[f"{REF_COLS[0]}_mu"]).sort_values("ym")
            sub = base.iloc[np.flatnonzero(miss)].sort_values("ym")
            f = pd.merge_asof(sub, r2, on="ym", by=GRP, direction="backward").set_index("_i")
            mm = mm.set_index("_i")
            for c in REF_COLS:
                for k in ("mu", "sd"):
                    col = f"{c}_{k}"
                    mm.loc[f.index, col] = f[col].to_numpy()
            mm = mm.reset_index()
    for c in REF_COLS:
        t[f"{c}z"] = (t[c].to_numpy() - mm[f"{c}_mu"].to_numpy()) / mm[f"{c}_sd"].to_numpy()
    return t


def add_reference_z(t: pd.DataFrame, window_months: int = 12, min_n: int = 200) -> pd.DataFrame:
    """(種別, 所属, 強度) ごとに、調教日より前の過去 window_months か月の平均・標準偏差で t4/t3/t1/accel を標準化する（小さい = 速い）。"""
    return apply_reference(t, build_reference(t, window_months, min_n))


def _pick(t: pd.DataFrame, lo: int, hi: int, prefix: str) -> pd.DataFrame:
    s = t[t["timed"] & t["days_before"].between(lo, hi)].sort_values(["horse_id", "race_date", "days_before"])
    g = s.groupby(["horse_id", "race_date"], sort=False).head(1)
    cols = {"days_before": "days", "intensity": "int", "t4z": "t4z", "t1z": "t1z", "accelz": "accelz", "accel": "accel", "t4": "t4", "t1": "t1", "kind": "kind", "center": "center", "position": "pos", "rank_n": "rank"}
    g = g[["horse_id", "race_date"] + list(cols)].rename(columns={k: f"{prefix}_{v}" for k, v in cols.items()})
    g[f"{prefix}_kind"] = g[f"{prefix}_kind"].map({"slope": 0, "wood": 1, "poly": 2}).astype("float32")
    g[f"{prefix}_center"] = g[f"{prefix}_center"].map({"栗": 0, "美": 1}).astype("float32")
    return g


def race_features(t: pd.DataFrame) -> pd.DataFrame:
    """(horse_id, race_date) ごとの調教の特徴。t は add_reference_z 済み。"""
    key = ["horse_id", "race_date"]
    F = _pick(t, 1, 6, "f")                   # 最終追い切り（レース週）
    W1 = _pick(t, 7, 13, "w1")                # 1 週前
    W2 = _pick(t, 14, 20, "w2")               # 2 週前
    base = t[key].drop_duplicates()
    out = base.merge(F, on=key, how="left").merge(W1, on=key, how="left").merge(W2, on=key, how="left")
    tm = t[t["timed"]]
    agg = tm[tm["days_before"] <= 14].groupby(key).agg(n_work_14=("t1", "size"), n_hard_14=("intensity", lambda s: int((s >= 1).sum())), mean_t4z_14=("t4z", "mean"), best_t1z_14=("t1z", "min"))
    agg28 = tm[tm["days_before"] <= 28].groupby(key).agg(n_work_28=("t1", "size"), slope_share_28=("kind", lambda s: float((s == "slope").mean())), best_t1z_28=("t1z", "min"))
    out = out.merge(agg.reset_index(), on=key, how="left").merge(agg28.reset_index(), on=key, how="left")
    for c in ("n_work_14", "n_hard_14", "n_work_28"):
        out[c] = out[c].fillna(0)
    last_any = t.groupby(key)["days_before"].min().rename("last_any_days").reset_index()
    out = out.merge(last_any, on=key, how="left")
    out["f_untimed_later"] = ((out["last_any_days"] < out["f_days"]) | (out["f_days"].isna() & out["last_any_days"].notna())).astype("float32")
    out["w1_exists"] = out["w1_days"].notna().astype("float32")
    out["w2_exists"] = out["w2_days"].notna().astype("float32")
    out["has_training"] = 1.0
    return out.drop(columns=["last_any_days"])


def add_self_relative(feat: pd.DataFrame) -> pd.DataFrame:
    """馬自身の過去レースの最終追い切りの標準化時計の平均との差（as-of: 過去のレースだけ）。"""
    f = feat.sort_values(["horse_id", "race_date"]).copy()
    for c in ("f_t4z", "f_t1z", "f_accelz"):
        prev = f.groupby("horse_id")[c].transform(lambda s: s.shift(1).expanding().mean())
        f[f"{c}_vs_self"] = f[c] - prev
    return f.sort_index()


# ---------------------------------------------------------------- 調教師別のパターン採点

def pattern_key(f: pd.DataFrame, accel_cut: float = 0.4) -> pd.Series:
    """最終追い切りの種別 × 強度 × 終いの形 × 1 週前の強さ × 最終の日数帯。計算できない行は空文字。"""
    fin = np.where(f["f_accel"] <= -accel_cut, "A", np.where(f["f_accel"] >= accel_cut, "D", "F"))
    w1 = np.where(f["w1_int"].isna(), "n", np.where(f["w1_int"] >= 1, "h", "e"))
    db = np.where(f["f_days"] <= 2, "s", np.where(f["f_days"] <= 4, "m", "l"))
    k = (f["f_kind"].fillna(-1).astype(int).astype(str) + "|" + f["f_int"].fillna(-1).astype(int).astype(str) + "|" + pd.Series(fin, index=f.index) + "|" + pd.Series(w1, index=f.index) + "|" + pd.Series(db, index=f.index))
    return k.where(f["f_accel"].notna(), "")


def trainer_pattern_scores(races: pd.DataFrame, m: float = 30.0) -> pd.DataFrame:
    """races: date・trainer_id・pattern・finish_position（出走ごと）。日付順に、前日までの結果だけで調教師×パターンの採点を作る。
    tp_top3 = (調教師×パターン)の複勝率をパターン全体の率へ縮約、tp_edge = それ − 調教師全体の複勝率（縮約）、pat_edge = パターン全体 − 全体、
    tp_n = 調教師×パターンの件数、tp_share = その調教師の出走のうち同パターンの割合（小さい = その厩舎では珍しい型）。win 版も同様。"""
    r = races.reset_index(drop=True)
    d = pd.to_datetime(r["date"]).to_numpy()
    top3 = (r["finish_position"].to_numpy() <= 3).astype(float)
    win = (r["finish_position"].to_numpy() == 1).astype(float)
    tid = r["trainer_id"].astype(str).to_numpy()
    pat = r["pattern"].to_numpy()
    out = np.full((len(r), 8), np.nan)
    order = np.argsort(d, kind="stable")
    tp: dict[tuple, list] = {}      # (trainer, pattern) -> [n, top3, win]
    tr: dict[str, list] = {}        # trainer -> [n, top3, win]
    pg: dict[str, list] = {}        # pattern -> [n, top3, win]
    gl = [0.0, 0.0, 0.0]
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and d[order[j]] == d[order[i]]:
            j += 1
        day = order[i:j]
        for r_ in day:
            if not pat[r_]:
                continue
            g_n, g_t3, g_w = gl
            if g_n < 500:
                continue
            base3, basew = g_t3 / g_n, g_w / g_n
            P = pg.get(pat[r_], [0, 0, 0])
            p3 = (P[1] + m * base3) / (P[0] + m)
            pw = (P[2] + m * basew) / (P[0] + m)
            T = tr.get(tid[r_], [0, 0, 0])
            t3 = (T[1] + m * base3) / (T[0] + m)
            tw = (T[2] + m * basew) / (T[0] + m)
            C = tp.get((tid[r_], pat[r_]), [0, 0, 0])
            c3 = (C[1] + m * p3) / (C[0] + m)
            cw = (C[2] + m * pw) / (C[0] + m)
            out[r_] = [c3, c3 - t3, p3 - base3, C[0], C[0] / max(T[0], 1), cw, cw - tw, pw - basew]
        for r_ in day:                                          # 当日の結果は翌日以降にだけ反映
            if not pat[r_]:
                continue
            for dct, k in ((tp, (tid[r_], pat[r_])), (tr, tid[r_]), (pg, pat[r_])):
                v = dct.setdefault(k, [0, 0, 0])
                v[0] += 1; v[1] += top3[r_]; v[2] += win[r_]
            gl[0] += 1; gl[1] += top3[r_]; gl[2] += win[r_]
        i = j
    return pd.DataFrame(out, columns=["tp_top3", "tp_edge", "pat_edge", "tp_n", "tp_share", "tp_win", "tp_win_edge", "pat_win_edge"], index=races.index)


# ---------------------------------------------------------------- 調教師内の相対化

TRN_COLS = ["f_t4z", "f_t1z", "f_accelz", "f_days", "f_int", "w1_exists", "n_work_14"]


def trainer_relative_features(feat: pd.DataFrame, trainer_id: pd.Series, date: pd.Series, min_prior: int = 20) -> pd.DataFrame:
    """その調教師の過去の出走（前日までに確定した日付の分）の平均に対する差。小さい = その厩舎としては速い／軽い。
    feat は race_features 済み（TRN_COLS を含む）で、trainer_id・date と同じ index。調教師は trainer_id で区別する。"""
    d = pd.DataFrame({"trn": trainer_id.astype(str).to_numpy(), "date": pd.to_datetime(date).to_numpy()}, index=feat.index)
    cols = [c for c in TRN_COLS if c in feat.columns]
    for c in cols:
        d[c] = feat[c].astype("float64").to_numpy()
    agg = d.groupby(["trn", "date"])[cols].agg(["sum", "count"])
    cs = agg.groupby(level=0).cumsum()
    prev = cs.groupby(level=0).shift(1)
    out = pd.DataFrame(index=feat.index)
    key = pd.MultiIndex.from_arrays([d["trn"], d["date"]])
    for c in cols:
        n = prev[(c, "count")].reindex(key).to_numpy()
        mean = prev[(c, "sum")].reindex(key).to_numpy() / np.where(n > 0, n, np.nan)
        out[f"{c}_vs_trn"] = np.where(n >= min_prior, feat[c].to_numpy(dtype="float64") - mean, np.nan)
        if c == cols[0]:
            out["trn_n_prior"] = np.nan_to_num(n)
    return out
