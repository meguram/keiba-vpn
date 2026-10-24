"""tp-77 追加5: 調教師ごとの『勝負追い切り』パターンの分析と特徴量（すべて開催日より前の結果だけから作る）。

python -m src.scripts.data.ptrain_workout build   # workout_rows.parquet（race_id, horse_id キー）と workout_report.json
観点（5つ）: 最終追い切り／前日追い切りの有無／最終の加速と4F時計の形／1週前追い切り／最終と1週前の種別の並び。
調教師ごとの『そのパターンの使用割合』（その調教師の過去の走から、全体の割合へ縮約）が低いものを勝負（普段と違う）、高いものを通常とする。
成績 res=(頭数−着順)/(頭数−1)−0.5。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.horse_history.common import FIT_END

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
TABLES = Path(os.environ.get("PTRAIN_TABLES", ROOT / "data/page_reference/tables"))
TL = os.environ.get("PTRAIN_TL", "")
OOF = Path(os.environ.get("PTRAIN_OOF", ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_z:RB@e.parquet"))
KEY = ["race_id", "horse_id"]
KAPPA_S, KAPPA_L = 20.0, 30.0
RARE, USUAL, MIN_T = 0.15, 0.5, 30        # そのパターンの使用回数 ÷ その調教師の最頻パターンの使用回数
LEGS = ["馬也", "一杯", "強め", "仕掛", "G強"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_runs() -> pd.DataFrame:
    cols = ["race_id", "date", "horse_id", "trainer_id", "trainer_name", "field_size", "finish_position"]
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in range(2019, 2026)], ignore_index=True)
    r["race_id"], r["horse_id"], r["trainer_id"] = r["race_id"].astype(str), r["horse_id"].astype(str), r["trainer_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"])
    for c in ("field_size", "finish_position"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["res"] = (r["field_size"] - r["finish_position"]) / (r["field_size"] - 1).replace(0, np.nan) - 0.5
    return r.dropna(subset=["res"])


VIEWS = ["final", "prevday", "shape", "week1", "route", "lap", "lapleg", "lapw1", "laproute"]


def _band(x: pd.Series, kind: pd.Series, fit: pd.Series) -> pd.Series:
    if not fit.any():
        fit = pd.Series(True, index=x.index)          # 固定期間にデータが無い場合（テスト用）
    q = x[fit].groupby(kind[fit]).quantile([1 / 3, 2 / 3]).unstack()
    lo, hi = kind.map(q[1 / 3]), kind.map(q[2 / 3])
    return pd.Series(np.where(x.isna() | lo.isna(), "?", np.where(x <= lo, "速", np.where(x <= hi, "中", "遅"))), index=x.index)


def _accel(a: pd.Series) -> pd.Series:
    return pd.Series(np.where(a.isna(), "?", np.where(a < -0.3, "加速", np.where(a > 0.3, "減速", "平行"))), index=a.index)


def lap_group(t1: pd.Series, t2: pd.Series, lmid: pd.Series | None = None) -> pd.Series:
    """終いのラップのグループ（ユーザー定義）。L1=終い1F、L2=その前の1F（t1,t2 は終いからの累積）。
    A1=終い1Fのみ12秒台の加速、B1=2Fのみ12秒台の減速、A2=終い2F各12秒台の加速、B2=同・減速、A3=終い1Fが11秒台(以下)の加速、B3=2Fが11秒台(以下)の減速。"""
    l1, l2 = t1, t2 - t1
    if lmid is not None:
        l2 = l2.fillna(lmid)                 # ウッド・ポリは 2F の記録が無いので、終い前の 2 ラップの平均(lmid)で近似
    in12 = lambda x: (x >= 12.0) & (x < 13.0)
    in11 = lambda x: x < 12.0
    g = np.select(
        [l1.isna() | l2.isna(),
         in11(l1) & (l1 < l2), in11(l2) & (l1 > l2),
         in12(l1) & in12(l2) & (l1 < l2), in12(l1) & in12(l2) & (l1 > l2),
         in12(l1) & (l2 >= 13.0) & (l1 < l2), in12(l2) & (l1 >= 13.0) & (l1 > l2)],
        ["?", "A3", "B3", "A2", "B2", "A1", "B1"], default="他")
    return pd.Series(g, index=t1.index)


def patterns(w: pd.DataFrame) -> pd.DataFrame:
    """走（horse_id, race_date）ごとに、調教師が見せ方を変えうる 5 つの観点のパターンを作る。
    final=最終追い切り(1〜6日前で最新)の 種別×脚色×終い1Fの時計階級×3週間の本数帯、
    prevday=前日(1日前)の追い切りの有無と種別、shape=最終追い切りの 種別×終いの加速(加速/平行/減速)×4F全体時計の階級、
    week1=1週前(7〜13日前)の 種別×脚色×加速、route=最終→1週前の種別の並び。"""
    w = w[w["days_before"].between(1, 20)].copy()
    w["race_date"] = pd.to_datetime(w["race_date"])
    fit = w["race_date"] <= pd.Timestamp(FIT_END)
    w["zb"] = _band(w["l1"], w["kind"], fit)
    w["tb"] = _band(w["t4"], w["kind"], fit)
    w["ac"] = _accel(w["accel"])
    w["lg"] = lap_group(w["t1"], w["t2"], w["lmid"])
    w["leg"] = w["leg_color"].where(w["leg_color"].isin(LEGS), "他")
    base = w.groupby(["horse_id", "race_date"]).agg(n3w=("kind", "size"), prev=("days_before", lambda s: bool((s == 1).any()))).reset_index()
    pk = w[w["days_before"] == 1].sort_values(["horse_id", "race_date"]).drop_duplicates(["horse_id", "race_date"])[["horse_id", "race_date", "kind"]].rename(columns={"kind": "pk"})
    f = w[w["days_before"] <= 6].sort_values(["horse_id", "race_date", "days_before"]).drop_duplicates(["horse_id", "race_date"])
    f = f[["horse_id", "race_date", "kind", "leg", "zb", "tb", "ac", "lg"]].add_prefix("f_").rename(columns={"f_horse_id": "horse_id", "f_race_date": "race_date"})
    w1 = w[w["days_before"].between(7, 13)].sort_values(["horse_id", "race_date", "days_before"]).drop_duplicates(["horse_id", "race_date"])
    w1 = w1[["horse_id", "race_date", "kind", "leg", "ac", "lg"]].add_prefix("w_").rename(columns={"w_horse_id": "horse_id", "w_race_date": "race_date"})
    d = base.merge(pk, on=["horse_id", "race_date"], how="left").merge(f, on=["horse_id", "race_date"], how="left").merge(w1, on=["horse_id", "race_date"], how="left")
    nb = np.where(d["n3w"] <= 2, "少", np.where(d["n3w"] <= 4, "並", "多"))
    hasf, hasw = d["f_kind"].notna(), d["w_kind"].notna()
    d["v_final"] = np.where(hasf, d["f_kind"].astype(str) + "|" + d["f_leg"].astype(str) + "|" + d["f_zb"].astype(str) + "|" + nb, "F無")
    d["v_prevday"] = np.where(d["prev"], "前日有|" + d["pk"].astype(str), "前日無")
    d["v_shape"] = np.where(hasf, d["f_kind"].astype(str) + "|" + d["f_ac"].astype(str) + "|" + d["f_tb"].astype(str), "F無")
    d["v_week1"] = np.where(hasw, d["w_kind"].astype(str) + "|" + d["w_leg"].astype(str) + "|" + d["w_ac"].astype(str), "W1無")
    d["v_route"] = np.where(hasf, d["f_kind"].astype(str), "F無") + "←" + np.where(hasw, d["w_kind"].astype(str), "W1無")
    d["v_lap"] = np.where(hasf, d["f_kind"].astype(str) + "|" + d["f_lg"].astype(str), "F無")
    d["v_lapleg"] = np.where(hasf, d["f_kind"].astype(str) + "|" + d["f_lg"].astype(str) + "|" + d["f_leg"].astype(str), "F無")
    d["v_lapw1"] = np.where(hasw, d["w_kind"].astype(str) + "|" + d["w_lg"].astype(str), "W1無")
    d["v_laproute"] = np.where(hasf, d["f_lg"].astype(str), "F無") + "←" + np.where(hasw, d["w_lg"].astype(str), "W1無")
    return d[["horse_id", "race_date", "n3w"] + [f"v_{v}" for v in VIEWS]]


def _cum(d: pd.DataFrame, keys: list[str], val: str | None = None) -> pd.DataFrame:
    g = d.groupby(keys + ["date"], sort=True)
    a = g.size().rename("n").reset_index() if val is None else g[val].agg(n="size", s="sum").reset_index()
    for c in [c for c in ("n", "s") if c in a.columns]:
        a[c] = a.groupby(keys)[c].cumsum() - a[c]
    return a


def _trainer_habit(R: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """調教師ごとに、より前の開催日までの使用回数から、(パターンの使用割合, 最頻パターンに対する比, 過去の走数) を返す（同日は含めない）。R は日付順。"""
    share, ratio, n_t = np.full(len(R), np.nan), np.full(len(R), np.nan), np.zeros(len(R))
    dates, pats = R["date"].to_numpy(), R["pat"].to_numpy()
    for _, idx in R.groupby("trainer_id", sort=False).indices.items():
        cnt: dict = {}
        n = mx = 0
        i = 0
        while i < len(idx):
            j = i
            while j < len(idx) and dates[idx[j]] == dates[idx[i]]:
                j += 1
            for k in idx[i:j]:
                n_t[k] = n
                if n > 0:
                    c = cnt.get(pats[k], 0)
                    share[k], ratio[k] = (c + 0.1) / (n + 1), c / mx
            for k in idx[i:j]:
                c = cnt.get(pats[k], 0) + 1
                cnt[pats[k]] = c
                mx = max(mx, c)
            n += j - i
            i = j
    return share, ratio, n_t


def asof_workout(R: pd.DataFrame, pat: str = "pat", pre: str = "P77_K") -> pd.DataFrame:
    """R: race_id, horse_id, date, trainer_id, res, pat 列, n3w（パターンのある走だけ）。pat 列名のパターンについて、列名は pre で始める。"""
    R = R.reset_index(drop=True)
    R = R.assign(pat=R[pat])
    share, ratio, n_t = _trainer_habit(R)
    ok = n_t >= MIN_T
    flag = np.where(~ok, "不明", np.where(ratio < RARE, "勝負", np.where(ratio >= USUAL, "通常", "中")))
    R["flag"] = flag
    out = pd.DataFrame({f"{pre}_share": share, f"{pre}_ratio": ratio, f"{pre}_rarity": -np.log(share), f"{pre}_n_t": n_t})
    out[f"{pre}_rare"] = (flag == "勝負").astype(float)
    ta = _cum(R, ["trainer_id"], "res").rename(columns={"n": "n_a", "s": "s_a"})
    tf = _cum(R, ["trainer_id", "flag"], "res").rename(columns={"n": "n_f", "s": "s_f"})
    X = R[["trainer_id", "flag", "date"]].merge(ta, on=["trainer_id", "date"], how="left").merge(tf, on=["trainer_id", "flag", "date"], how="left")
    ma = X["s_a"].fillna(0) / (X["n_a"].fillna(0) + KAPPA_L)
    out[f"{pre}_lift_this"] = (X["s_f"].fillna(0) / (X["n_f"].fillna(0) + KAPPA_L) - ma).to_numpy()
    for fl, nm in (("勝負", "rare"), ("通常", "usual")):
        Y = R[["trainer_id", "date"]].assign(flag=fl).merge(tf, on=["trainer_id", "flag", "date"], how="left")
        out[f"{pre}_lift_{nm}"] = (Y["s_f"].fillna(0) / (Y["n_f"].fillna(0) + KAPPA_L) - ma).to_numpy()
    out[f"{pre}_lift_gap"] = out[f"{pre}_lift_rare"] - out[f"{pre}_lift_usual"]
    out.index = R.index
    out[f"{pre}_flag"] = flag
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p); m = len(p); q = np.empty(m); prev = 1.0
    for rank, i in zip(range(m, 0, -1), o[::-1]):
        prev = min(prev, p[i] * m / rank); q[i] = prev
    return q


def report(R: pd.DataFrame, flag: pd.Series, oof: pd.DataFrame | None) -> dict:
    from math import erfc, sqrt

    R = R.assign(flag=flag.to_numpy())
    R = R[R["date"] >= "2020-01-01"].copy()
    R["year"] = R["date"].dt.year
    rep: dict = {}
    g = R.groupby("flag").agg(n_runs=("res", "size"), n_races=("race_id", "nunique"), mean_res=("res", "mean")).round(4)
    ys = R.groupby(["flag", "year"])["res"].mean().unstack()
    g["years_below_usual"] = [int((ys.loc[f] < ys.loc["通常"]).sum()) if "通常" in ys.index else None for f in g.index]
    rep["by_flag"] = g.reset_index().to_dict("records")
    if "通常" not in ys.index:
        return rep
    if oof is not None:
        d = R.merge(oof, on=KEY, how="inner")
        d["y3"] = (d["finish_position"] <= 3).astype(float); d["y1"] = (d["finish_position"] == 1).astype(float)
        d["r3"] = d["y3"] - d["p_top3"]; d["r1"] = d["y1"] - d["p_win"]
        rep["by_flag_vs_v6e"] = d.groupby("flag").agg(n=("r3", "size"), resid_top3=("r3", "mean"), resid_win=("r1", "mean")).round(4).reset_index().to_dict("records")
    rows = []
    for (t, nm), x in (R.groupby(["trainer_id", "trainer_name"]) if "勝負" in set(R["flag"]) else []):
        a, b = x[x["flag"] == "勝負"]["res"], x[x["flag"] == "通常"]["res"]
        if len(x) < 400 or len(a) < 40 or len(b) < 40:
            continue
        se = sqrt(a.var() / len(a) + b.var() / len(b)); z = (a.mean() - b.mean()) / se
        rows.append({"trainer": nm, "n_rare": len(a), "n_usual": len(b), "diff": round(float(a.mean() - b.mean()), 4), "z": round(float(z), 2), "p": erfc(abs(z) / sqrt(2))})
    if rows:
        T = pd.DataFrame(rows); T["q"] = _bh(T["p"].to_numpy())
        rep["trainers"] = {"n_tested": len(T), "n_q_lt_0.1": int((T["q"] < 0.1).sum()), "expected_under_null": round(0.1 * len(T), 1),
                           "top": T.nsmallest(15, "p").round(4).to_dict("records")}
    return rep


def cmd_build(a) -> None:
    path = TL or glob.glob(str(ROOT / "data/local/**/training_laps.parquet"), recursive=True)[0]
    w = pd.read_parquet(path, columns=["horse_id", "race_date", "kind", "leg_color", "l1", "accel", "t1", "t2", "lmid", "t4", "days_before"])
    w["horse_id"] = w["horse_id"].astype(str)
    pat = patterns(w)
    r = load_runs()
    R = r.merge(pat, left_on=["horse_id", "date"], right_on=["horse_id", "race_date"], how="inner").drop(columns=["race_date"])
    _log(f"パターンのある走 {len(R)} / {len(r)}")
    R = R.sort_values(["date", "race_id"]).reset_index(drop=True)
    parts, flags = [R[KEY], R[["n3w"]].rename(columns={"n3w": "P77_K_n3w"}).astype("float32")], {}
    for i, v in enumerate(VIEWS):
        f = asof_workout(R, f"v_{v}", f"P77_K_{v}")
        flags[v] = f.pop(f"P77_K_{v}_flag")
        if i:
            f = f.drop(columns=[f"P77_K_{v}_n_t"])
        parts.append(f.astype("float32"))
    out = pd.concat(parts, axis=1)
    out["P77_K_n_views_rare"] = sum(out[f"P77_K_{v}_rare"] for v in VIEWS).astype("float32")
    full = r[KEY].merge(out, on=KEY, how="left")
    full.to_parquet(OUT / "workout_rows.parquet", index=False)
    oof = None
    if OOF.exists():
        oof = pd.read_parquet(OOF)
        oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
        oof = oof[KEY + [c for c in ("p_win", "p_top3") if c in oof.columns]]
    rep = {v: report(R, flags[v], oof) for v in VIEWS}
    rep["n_views_rare"] = report(R, out["P77_K_n_views_rare"].astype(int).astype(str).reset_index(drop=True).rename("flag"), oof).get("by_flag")
    (OUT / "workout_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"workout_rows: {full.shape}、被覆 {float(full['P77_K_final_share'].notna().mean()):.3f}、"
         + "、".join(f"{v}:勝負 {float(out[f'P77_K_{v}_rare'].mean()):.3f}" for v in VIEWS))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
