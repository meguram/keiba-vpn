"""tp-59 (2)(3): 騎手・調教師の勝率・複勝率・単勝/複勝回収率、市場への織り込み、年ごとの再現性。

python -m src.scripts.data.tp59_profile run     # ホストで実行。<TP59_OUT>/profile.json
入力: <TP59_OUT>/master.parquet（tp59_data build）。2026 年は選択に使わず、報告だけ（年組 2025→2026 は『確認』と明記）。

回収率は『1 出走 100 円』。区間はレース単位ブートストラップ。関係者別の指標は ID で集計し、件数の下限（既定 100 出走/年、2026 年は部分年なので 70）を置く。
超過払戻（ex_*）は、単勝オッズの帯ごとの平均払戻を引いたもの（人気馬ばかり乗る関係者が高く見えるだけ、を除く）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp59_lib as L

OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
SEL_YEARS = (2022, 2023, 2024)
N_MIN = 100
N_MIN_2026 = 70
K_WIN = 40.0          # 縮約の強さ（出走数換算）
K_ROI = 200.0


def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "master.parquet")
    d = d[d["jockey_id"].notna() & d["trainer_id"].notna()].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["p_win_raw"] = d["p_win"]
    d["p_win"] = d["p_win"].fillna(d["q_win"]).where(d["year"] >= 2022)
    return L.add_excess(d)


def n_min_for(y: int) -> int:
    return N_MIN_2026 if y == 2026 else N_MIN


def metrics(t: pd.DataFrame) -> pd.DataFrame:
    """関係者 × 年 の合計表から率・回収率・超過・市場との差を出す。"""
    t = t.copy()
    t["win_rate"] = t["win"] / t["n"]
    t["top2_rate"] = t["top2"] / t["n"]
    t["top3_rate"] = t["top3"] / t["n"]
    t["roi_win"] = t["ret_win"] / (100.0 * t["n"])
    t["roi_place"] = t["ret_place"] / (100.0 * t["n"])
    t["ex_roi_win"] = t["ex_win"] / (100.0 * t["n"])
    t["ex_roi_place"] = t["ex_place"] / (100.0 * t["n"])
    t["resid_q"] = (t["win"] - t["q_win"]) / t["n"]
    if "p_win" in t.columns:
        t["resid_p"] = (t["win"] - t["p_win"]) / t["n"]
    return t


METRICS = ["win_rate", "top3_rate", "roi_win", "roi_place", "ex_roi_win", "ex_roi_place", "resid_q"]


def population(d: pd.DataFrame) -> dict:
    out = {"by_year": {}, "by_odds_band": {}}
    for y, g in d.groupby("year"):
        out["by_year"][int(y)] = {"win": L.boot_roi(g["race_id"].to_numpy(), g["ret_win"].to_numpy()), "place": L.boot_roi(g["race_id"].to_numpy(), g["ret_place"].to_numpy()),
                                  "win_rate": float(g["win"].mean()), "top3_rate": float(g["top3"].mean())}
    periods = {"2022-2024": d[d["year"].isin(SEL_YEARS)], "2025": d[d["year"] == 2025], "2026": d[d["year"] == 2026]}
    for name, g in periods.items():
        rows = []
        for b, gg in g.groupby("obs_band"):
            rows.append({"band": b, "win": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_win"].to_numpy(), 500), "place": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_place"].to_numpy(), 500)})
        out["by_odds_band"][name] = rows
    return out


def entity_persistence(d: pd.DataFrame, key: str) -> dict:
    """関係者（key = jockey_id / trainer_id）の、前年 → 翌年の再現性。"""
    ty = L.entity_year_table(d, [key])
    ty = metrics(ty)
    res: dict = {"reliability": {}, "pairs": {}, "top_decile": {}, "top_k": {}}
    for y in range(2020, 2027):
        x = ty[ty["year"] == y]
        res["reliability"][int(y)] = {"win": L.reliability(x, "win", n_min_for(y)), "place": L.reliability(x, "place", n_min_for(y)),
                                       "entities": int((x["n"] >= n_min_for(y)).sum())}
    for y in range(2020, 2026):
        a = ty[(ty["year"] == y) & (ty["n"] >= N_MIN)].set_index(key)
        b = ty[(ty["year"] == y + 1) & (ty["n"] >= n_min_for(y + 1))].set_index(key)
        com = a.index.intersection(b.index)
        row = {"entities": int(len(com))}
        for m in METRICS + (["resid_p"] if y >= 2022 and "resid_p" in ty.columns else []):
            row[m] = L.spearman(a.loc[com, m].to_numpy(), b.loc[com, m].to_numpy())
        res["pairs"][f"{y}->{y+1}"] = row
        # 前年の成績が上位 10% の関係者の全出走を翌年に買う（前年に 100 出走以上の関係者だけを候補とする）
        nxt = d[(d["year"] == y + 1) & d[key].isin(a.index)]
        cell: dict = {}
        for m in ("roi_win", "roi_place", "ex_roi_win", "ex_roi_place", "win_rate", "top3_rate"):
            thr = a[m].quantile(0.9)
            top = a.index[a[m] >= thr]
            sel = nxt[nxt[key].isin(top)]
            if len(sel) == 0:
                continue
            cell[m] = {}
            for bet, col in (("win", "ret_win"), ("place", "ret_place")):
                r = L.boot_roi(sel["race_id"].to_numpy(), sel[col].to_numpy(), 500)
                df_ = L.boot_diff(sel["race_id"].to_numpy(), sel[col].to_numpy(), nxt["race_id"].to_numpy(), nxt[col].to_numpy(), 500)
                cell[m][bet] = {"top": r, "all_eligible_roi": float(100 * nxt[col].sum() / (100.0 * len(nxt))), "diff": df_, "entities_top": int(len(top))}
        res["top_decile"][f"{y}->{y+1}"] = cell
    return res


def popularity_effect(d: pd.DataFrame, key: str) -> list[dict]:
    """前年の出走数順位（上位 10 / 11〜30 / それ以外）の関係者の翌年の成績。人気騎手・厩舎は買われすぎか。"""
    rows = []
    for y in range(2021, 2027):
        prev = d[d["year"] == y - 1].groupby(key).size().sort_values(ascending=False)
        grp = pd.Series("rest", index=prev.index)
        grp.iloc[:10] = "top10"
        grp.iloc[10:30] = "11-30"
        g = d[d["year"] == y].copy()
        g["grp"] = g[key].map(grp).fillna("new")
        for k, gg in g.groupby("grp"):
            rows.append({"year": int(y), "grp": k, "n": int(len(gg)), "avg_win_odds": float(gg["win_odds"].median()), "win_rate": float(gg["win"].mean()), "q_sum_rate": float(gg["q_win"].mean()),
                         "win": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_win"].to_numpy(), 300), "place": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_place"].to_numpy(), 300),
                         "ex_win_roi": float(gg["ex_win"].sum() / (100.0 * len(gg))), "ex_place_roi": float(gg["ex_place"].sum() / (100.0 * len(gg)))})
    return rows


def add_strength(d: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    """as-of（当年より前）の縮約した勝率を、その関係者の強さとして各出走に付ける。"""
    ty = L.entity_year_table(d, [key])
    d = d.copy()
    d[name] = np.nan
    prev_name = f"prev_{name}"
    d[prev_name] = np.nan
    for y in sorted(d["year"].unique()):
        a = L.asof_table(ty, [key], y)
        if a.empty:
            continue
        r0 = a["win"].sum() / a["n"].sum()
        s = ((a["win"] + K_WIN * r0) / (a["n"] + K_WIN)).set_axis(a[key])
        m = d["year"] == y
        d.loc[m, name] = d.loc[m, key].map(s).fillna(r0)
        if key == "jockey_id":
            d.loc[m, prev_name] = d.loc[m, "prev_jockey_id"].map(s)
    return d


def condition_tables(d: pd.DataFrame) -> dict:
    d = add_strength(add_strength(d, "jockey_id", "j_str"), "trainer_id", "t_str")
    d = d[d["year"] >= 2022].copy()
    d["j_tier"] = d.groupby("year")["j_str"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=["j1_low", "j2", "j3", "j4", "j5_high"]).astype(str))
    d["t_tier"] = d.groupby("year")["t_str"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=["t1_low", "t2", "t3", "t4", "t5_high"]).astype(str))
    up = d["j_str"] - d["prev_j_str"]
    d["jockey_move"] = np.select([d["chg_jockey"] == 0, (d["chg_jockey"] == 1) & (up > 0.01), (d["chg_jockey"] == 1) & (up < -0.01), d["chg_jockey"] == 1],
                                 ["same", "up", "down", "flat"], "first")
    d["lay_band"] = pd.cut(d["lay_days"], [-1, 14, 35, 90, 180, 1e9], labels=["~14d", "15-35d", "36-90d", "91-180d", "181d+"]).astype(str).replace("nan", "first")
    d["prev_fin_band"] = pd.cut(d["prev_fin"], [0, 1, 3, 5, 9, 99], labels=["1", "2-3", "4-5", "6-9", "10+"]).astype(str).replace("nan", "first")
    d["age"] = pd.to_numeric(d["sex_age"].astype(str).str[1:], errors="coerce").clip(upper=6)
    d["sex"] = d["sex_age"].astype(str).str[:1]
    d["chg_trainer_f"] = d["chg_trainer"].map({0.0: "same", 1.0: "changed"}).fillna("first")
    d["surf_dist"] = d["surface"].astype(str) + "_" + d["dist_band"].astype(str)
    out: dict = {}
    periods = {"2022-2024": d[d["year"].isin(SEL_YEARS)], "2025": d[d["year"] == 2025], "2026": d[d["year"] == 2026]}
    for col in ["jockey_move", "j_tier", "t_tier", "lay_band", "prev_fin_band", "sex", "age", "cls", "surf_dist", "venue_code", "chg_trainer_f"]:
        out[col] = {}
        for name, g in periods.items():
            rows = []
            for k, gg in g.groupby(col, observed=True):
                if len(gg) < 300:
                    continue
                rows.append({"level": str(k), "n": int(len(gg)), "win": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_win"].to_numpy(), 300),
                             "place": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_place"].to_numpy(), 300),
                             "ex_win": float(gg["ex_win"].sum() / (100.0 * len(gg))), "ex_place": float(gg["ex_place"].sum() / (100.0 * len(gg))),
                             "win_rate": float(gg["win"].mean()), "top3_rate": float(gg["top3"].mean()), "q_rate": float(gg["q_win"].mean()),
                             "p_rate": float(gg["p_win_raw"].mean()) if gg["p_win_raw"].notna().any() else None})
            out[col][name] = rows
    # 騎手の強さ × オッズ帯の格子（2022〜2024 の合算。回収率）
    grid = []
    g = periods["2022-2024"]
    for (jt, ob), gg in g.groupby(["j_tier", "obs_band"]):
        if len(gg) < 300:
            continue
        grid.append({"j_tier": jt, "band": ob, "n": int(len(gg)), "win_roi": float(gg["ret_win"].mean()), "place_roi": float(gg["ret_place"].mean()),
                     "win_rate": float(gg["win"].mean()), "q_rate": float(gg["q_win"].mean())})
    out["jtier_x_odds"] = grid
    return out


def calibration_by_entity(d: pd.DataFrame) -> dict:
    """V4 の勝率が、関係者の強さ別に偏っていないか（勝率の実績 − モデル確率、実績 − 市場確率）。"""
    d = add_strength(add_strength(d, "jockey_id", "j_str"), "trainer_id", "t_str")
    d = d[d["p_win_raw"].notna() & (d["year"] >= 2022)].copy()
    out = {}
    for col, nm in (("j_str", "jockey"), ("t_str", "trainer")):
        d["tier"] = d.groupby("year")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
        rows = []
        for (per, g) in (("2022-2024", d[d["year"].isin(SEL_YEARS)]), ("2025", d[d["year"] == 2025]), ("2026", d[d["year"] == 2026])):
            for k, gg in g.groupby("tier"):
                rows.append({"period": per, "decile": int(k), "n": int(len(gg)), "win_rate": float(gg["win"].mean()), "model_p": float(gg["p_win_raw"].mean()), "market_q": float(gg["q_win"].mean()),
                             "roi_win": float(gg["ret_win"].mean())})
        out[nm] = rows
    return out


def main() -> None:
    d = load()
    res = {"population": population(d)}
    for key, nm in (("jockey_id", "jockey"), ("trainer_id", "trainer")):
        res[f"persist_{nm}"] = entity_persistence(d, key)
        res[f"popularity_{nm}"] = popularity_effect(d, key)
        print(nm, "done", flush=True)
    res["conditions"] = condition_tables(d)
    res["calibration"] = calibration_by_entity(d)
    (OUT / "profile.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print("saved", OUT / "profile.json")


if __name__ == "__main__":
    main()
