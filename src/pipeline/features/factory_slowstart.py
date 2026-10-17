"""特徴量ファクトリー G6「出遅れ率」。

ラベル（出遅れたか）の入手元は 2 つ:
  * 実ラベル: 戦績ページの ｽﾀｰﾄ指数・備考（カテゴリ horse_race_remarks。取り込み中）。``real_labels``。
  * 代理ラベル: SmartRC のテン1F・最初のコーナー位置が、その馬自身の過去（as-of）から大きく外れて遅い走。``proxy_labels``。

どちらも「馬 × 日付」の表（``lab``）にそろえ、そこから全て as-of（当該レース日より前だけ）で特徴量を作る。
キーは horse_id / jockey_id / trainer_id / 種牡馬 ID。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
P0 = 0.08                      # 出遅れ率の公称値（全体の率の as-of 推定の出発点）


# ───────────────────────── ラベル
def proxy_labels(career: pd.DataFrame, norm_before: str = "20200101", q_a: float = 0.96, q_b: float = 0.92, q_drop: float = 0.80) -> pd.DataFrame:
    """career(horse_id, rdate, ten, c1_rel, trackkind, range) → 馬×日付のラベル。

    ten_z = 条件（trackkind×距離）で標準化したテン1F。resid = ten_z − その馬の過去走の平均 ten_z（過去 2 走以上）。c1_drop = 最初の角の位置(相対)の悪化。
    しきい値と標準化の平均・標準偏差は norm_before より前の走だけから決める（評価期間の情報を使わない）。"""
    c = career.dropna(subset=["horse_id", "rdate"]).drop_duplicates(["horse_id", "rdate"]).sort_values(["horse_id", "rdate"]).reset_index(drop=True)
    c["date"] = pd.to_datetime(c["rdate"].astype(str), format="%Y%m%d", errors="coerce")
    pre = c["rdate"].astype(str) < norm_before
    ok = c["ten"].notna() & pre
    g = c[ok].groupby(["trackkind", "range"])["ten"].agg(["mean", "std", "size"])
    g = g[g["size"] >= 30]
    glob = (c.loc[ok, "ten"].mean(), c.loc[ok, "ten"].std())
    idx = pd.MultiIndex.from_frame(c[["trackkind", "range"]])
    mu = pd.Series(g["mean"].reindex(idx).to_numpy(), index=c.index).fillna(glob[0])
    sd = pd.Series(g["std"].reindex(idx).to_numpy(), index=c.index).fillna(glob[1]).clip(lower=0.05)
    c["ten_z"] = (c["ten"] - mu) / sd
    h = c["horse_id"]
    for src, name in (("ten_z", "ten"), ("c1_rel", "c1")):
        v = c[src]
        S = v.fillna(0).groupby(h).cumsum() - v.fillna(0)          # 自分より前の合計
        N = v.notna().astype(int).groupby(h).cumsum() - v.notna().astype(int)
        c[f"base_{name}"] = (S / N.replace(0, np.nan)).where(N >= 2)
    c["resid"] = c["ten_z"] - c["base_ten"]
    c["c1_drop"] = c["c1_rel"] - c["base_c1"]
    prer = c.loc[pre, "resid"].dropna()
    t_a, t_b = prer.quantile(q_a), prer.quantile(q_b)
    t_d = c.loc[pre, "c1_drop"].dropna().quantile(q_drop)
    r = c["resid"]
    c["slow_a"] = (r > t_a).astype(float).where(r.notna())
    c["slow_b"] = (r > t_b).astype(float).where(r.notna())
    c["slow_c"] = ((r > t_b) & (c["c1_drop"] > t_d)).astype(float).where(r.notna() & c["c1_drop"].notna())
    out = c[["horse_id", "date", "resid", "c1_drop", "slow_a", "slow_b", "slow_c"]].copy()
    out.attrs["thresholds"] = {"resid_q_a": float(t_a), "resid_q_b": float(t_b), "c1_drop": float(t_d)}
    return out


def real_labels(rem: pd.DataFrame, start_q: float = 0.05, norm_before: str = "2023-01-01") -> pd.DataFrame:
    """rem(horse_id, race_id, date, start_index, remarks) → 馬×日付のラベル。slow_idx はｽﾀｰﾄ指数が低い走、slow_rem は備考に『出遅』。
    指数のしきい値は norm_before より前の走の分布から決める。"""
    r = rem.copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    si = pd.to_numeric(r["start_index"], errors="coerce")
    thr = si[r["date"] < pd.Timestamp(norm_before)].quantile(start_q)
    r["slow_idx"] = (si <= thr).astype(float).where(si.notna())
    rm = r["remarks"].fillna("").astype(str)
    r["slow_rem"] = rm.str.contains("出遅").astype(float)
    r["trouble_rem"] = rm.str.contains("不利|落馬|競走中止|躓|挟まれ").astype(float)
    r["start_z"] = (si - si.mean()) / si.std()
    out = r[["horse_id", "date", "start_index", "start_z", "slow_idx", "slow_rem", "trouble_rem"]].copy()
    out.attrs["thresholds"] = {"start_index_q": float(thr)}
    return out


# ───────────────────────── as-of の集計
def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    return k


def asof_rate(frame: pd.DataFrame, ev: pd.DataFrame, keys: list[str], val: str, m: float, prior, name: str) -> pd.DataFrame:
    """frame の各行について、同じキーの ev の行のうち日付が厳密に前のものの val の平均を、prior（定数か、frame と同じ長さの配列）へ m 件分縮約して返す。"""
    f = pd.DataFrame({"_k": _key(frame, keys).to_numpy(), "date": frame["date"].to_numpy(), "_i": np.arange(len(frame))}).sort_values("date")
    e = ev.dropna(subset=[val])
    e = pd.DataFrame({"_k": _key(e, keys).to_numpy(), "date": e["date"].to_numpy(), "v": e[val].to_numpy(dtype=float)})
    g = e.groupby(["_k", "date"], sort=False).agg(s=("v", "sum"), n=("v", "size")).reset_index().sort_values("date")
    g["S"] = g.groupby("_k")["s"].cumsum()
    g["N"] = g.groupby("_k")["n"].cumsum()
    j = pd.merge_asof(f, g[["_k", "date", "S", "N"]], on="date", by="_k", allow_exact_matches=False, direction="backward").sort_values("_i")
    n, s = j["N"].fillna(0).to_numpy(), j["S"].fillna(0).to_numpy()
    return pd.DataFrame({f"{name}_rate": (s + m * prior) / (n + m), f"{name}_n": n}, index=frame.index)


def horse_windows(frame: pd.DataFrame, lab: pd.DataFrame, val: str, prior, name: str, m: float = 5.0) -> pd.DataFrame:
    """馬ごとの出遅れ率: 全期間（縮約）・直近 5/10 走・指数平滑・件数・直近の連続。lab は馬×日付×val（過去分も含む）。"""
    d = lab.dropna(subset=[val]).sort_values(["horse_id", "date"]).copy()
    v = d[val].astype(float)
    grp = d["horse_id"]
    d["S"] = v.groupby(grp).cumsum()
    d["N"] = v.groupby(grp).cumcount() + 1
    d["r5"] = v.groupby(grp).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    d["r10"] = v.groupby(grp).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    d["ew"] = v.groupby(grp).ewm(alpha=0.3).mean().reset_index(level=0, drop=True)
    d["last"] = v
    f = pd.DataFrame({"horse_id": frame["horse_id"].to_numpy(), "date": frame["date"].to_numpy(), "_i": np.arange(len(frame))}).sort_values("date")
    j = pd.merge_asof(f, d[["horse_id", "date", "S", "N", "r5", "r10", "ew", "last"]].sort_values("date"), on="date", by="horse_id", allow_exact_matches=False, direction="backward").sort_values("_i")
    n = j["N"].fillna(0).to_numpy()
    out = pd.DataFrame({f"{name}_h_rate": (j["S"].fillna(0).to_numpy() + m * prior) / (n + m), f"{name}_h_n": n,
                        f"{name}_h_r5": j["r5"].to_numpy(), f"{name}_h_r10": j["r10"].to_numpy(), f"{name}_h_ew": j["ew"].to_numpy(),
                        f"{name}_h_last": j["last"].to_numpy()}, index=frame.index)
    return out


def damage(frame: pd.DataFrame, fin_pct: str, flag: str, m: float = 3.0) -> pd.DataFrame:
    """出遅れた走で着順(相対)がどれだけ悪化するか: 馬ごとの as-of 平均 (出遅れ走の着順% − 出遅れでない走の着順%)。frame は馬×日付の走（flag, fin_pct を持つ）。"""
    d = frame[["horse_id", "date", fin_pct, flag]].dropna().copy()
    d["x1"] = d[fin_pct].where(d[flag] > 0, 0.0)
    d["n1"] = (d[flag] > 0).astype(float)
    d["x0"] = d[fin_pct].where(d[flag] == 0, 0.0)
    d["n0"] = (d[flag] == 0).astype(float)
    res = {}
    for c in ("x1", "n1", "x0", "n0"):
        a = asof_cum_sum(frame, d, ["horse_id"], c)
        res[c] = a
    mean_all = 0.5                 # 着順の相対位置の平均は定義上 0.5（全期間の平均を使うと将来が入る）
    m1 = (res["x1"] + m * mean_all) / (res["n1"] + m)
    m0 = (res["x0"] + m * mean_all) / (res["n0"] + m)
    return pd.DataFrame({"damage": m1 - m0, "damage_n1": res["n1"]}, index=frame.index)


def asof_cum_sum(frame: pd.DataFrame, ev: pd.DataFrame, keys: list[str], val: str) -> np.ndarray:
    f = pd.DataFrame({"_k": _key(frame, keys).to_numpy(), "date": frame["date"].to_numpy(), "_i": np.arange(len(frame))}).sort_values("date")
    e = pd.DataFrame({"_k": _key(ev, keys).to_numpy(), "date": ev["date"].to_numpy(), "v": ev[val].to_numpy(dtype=float)})
    g = e.groupby(["_k", "date"], sort=False)["v"].sum().reset_index().sort_values("date")
    g["S"] = g.groupby("_k")["v"].cumsum()
    j = pd.merge_asof(f, g[["_k", "date", "S"]], on="date", by="_k", allow_exact_matches=False, direction="backward").sort_values("_i")
    return j["S"].fillna(0).to_numpy()


# ───────────────────────── 群ごとの特徴量
def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """文脈の列を足す（距離帯・枠の内外・斤量帯・性別/年齢・出走間隔）。frame は race_id, horse_id, date(datetime), horse_number, field_size,
    distance, surface, venue_code, sex_age, weight, bracket_number を持つ。"""
    f = frame.copy()
    f["dband"] = pd.cut(f["distance"].astype(float), [0, 1300, 1700, 2100, 2500, 9999], labels=False).fillna(-1).astype(int)
    fs = f["field_size"].astype(float).clip(lower=1)
    pos = (f["horse_number"].astype(float) - 1) / (fs - 1).clip(lower=1)
    f["gate3"] = np.where(pos < 1 / 3, "in", np.where(pos < 2 / 3, "mid", "out"))
    f["fband"] = pd.cut(fs, [0, 9, 13, 16, 99], labels=False).fillna(-1).astype(int)
    f["wband"] = pd.cut(f["weight"].astype(float), [0, 440, 480, 520, 9999], labels=False).fillna(-1).astype(int)
    sa = f["sex_age"].astype(str)
    f["sex"] = sa.str[:1]
    f["age"] = pd.to_numeric(sa.str[1:], errors="coerce").fillna(-1).clip(upper=8).astype(int)
    f = f.sort_values(["horse_id", "date"])
    gap = f.groupby("horse_id")["date"].diff().dt.days
    f["lband"] = pd.cut(gap, [0, 14, 35, 70, 140, 100000], labels=False).fillna(-1).astype(int)
    f["surf_dband"] = f["surface"].astype(str) + f["dband"].astype(str)
    return f.sort_index()


def build_groups(frame: pd.DataFrame, lab: pd.DataFrame, val: str = "slow_b", parents: pd.DataFrame | None = None, p0: float = P0) -> dict[str, pd.DataFrame]:
    """frame: prepare 済み。lab: 馬×日付のラベル（val 列）。parents: horse_id, sire_id, dam_sire_id。戻り値: 群名 → 特徴量表（frame と同じ index）。"""
    f = frame.merge(lab[["horse_id", "date", val]].drop_duplicates(["horse_id", "date"]), on=["horse_id", "date"], how="left", suffixes=("", "_lab")) if val not in frame.columns else frame
    f.index = frame.index
    if parents is not None:
        f = f.merge(parents[["horse_id", "sire_id", "dam_sire_id"]], on="horse_id", how="left")
        f.index = frame.index
    ev = f
    f = f.assign(_const="all")
    ev = f
    # 全体の率も as-of（全期間の平均を使うと将来のラベルが入る）。最初の日は P0 から始め、日が進むにつれて実測へ
    prior = asof_rate(f, ev, ["_const"], val, 200, p0, "g")["g_rate"].to_numpy()
    out: dict[str, pd.DataFrame] = {}
    out["SS_horse"] = horse_windows(f, lab, val, prior, "ss")
    h = []
    for nm, keys, m in (("h_surf", ["horse_id", "surface"], 5), ("h_dband", ["horse_id", "dband"], 5), ("h_venue", ["horse_id", "venue_code"], 5), ("h_gate3", ["horse_id", "gate3"], 5)):
        h.append(asof_rate(f, ev, keys, val, m, prior, "ss_" + nm))
    out["SS_horse_ctx"] = pd.concat(h, axis=1)
    g = []
    for nm, keys, m in (("g_tert", ["venue_code", "surface", "dband", "gate3"], 100), ("g_num", ["venue_code", "surface", "distance", "horse_number"], 60),
                        ("g_brk", ["bracket_number"], 200), ("g_fgate", ["fband", "gate3"], 200)):
        g.append(asof_rate(f, ev, keys, val, m, prior, "ss_" + nm))
    out["SS_gate"] = pd.concat(g, axis=1)
    out["SS_jockey"] = pd.concat([asof_rate(f, ev, ["jockey_id"], val, 30, prior, "ss_jk"), asof_rate(f, ev, ["jockey_id", "surface"], val, 30, prior, "ss_jk_surf"),
                                  asof_rate(f, ev, ["jockey_id", "gate3"], val, 30, prior, "ss_jk_gate")], axis=1)
    out["SS_trainer"] = pd.concat([asof_rate(f, ev, ["trainer_id"], val, 30, prior, "ss_tr"), asof_rate(f, ev, ["trainer_id", "surface"], val, 30, prior, "ss_tr_surf")], axis=1)
    if parents is not None:
        out["SS_sire"] = pd.concat([asof_rate(f, ev, ["sire_id"], val, 50, prior, "ss_sire"), asof_rate(f, ev, ["dam_sire_id"], val, 50, prior, "ss_dsire"),
                                    asof_rate(f, ev, ["sire_id", "surface"], val, 50, prior, "ss_sire_surf")], axis=1)
    out["SS_misc"] = pd.concat([asof_rate(f, ev, ["sex", "age"], val, 100, prior, "ss_sexage"), asof_rate(f, ev, ["lband"], val, 100, prior, "ss_layoff"),
                                asof_rate(f, ev, ["fband"], val, 100, prior, "ss_field"), asof_rate(f, ev, ["wband"], val, 100, prior, "ss_wt"),
                                asof_rate(f, ev, ["venue_code", "surface", "dband"], val, 100, prior, "ss_course")], axis=1)
    return out


def with_damage(frame: pd.DataFrame, lab: pd.DataFrame, val: str = "slow_b") -> pd.DataFrame:
    f = frame[["horse_id", "date", "finish_position", "field_size"]].merge(lab[["horse_id", "date", val]].drop_duplicates(["horse_id", "date"]), on=["horse_id", "date"], how="left")
    f.index = frame.index
    f["fin_pct"] = (f["finish_position"].astype(float) - 1) / (f["field_size"].astype(float) - 1).clip(lower=1)
    f.loc[~(f["finish_position"].astype(float) > 0), "fin_pct"] = np.nan
    return damage(f, "fin_pct", val)
