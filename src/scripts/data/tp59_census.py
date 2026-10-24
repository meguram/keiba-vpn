"""tp-59 (2) 補足: 『回収率が 100% を超える関係者（またはその組み合わせ）は偶然以上にいるか』の全数調査と、その後の年の成績。

python -m src.scripts.data.tp59_census run     # <TP59_OUT>/census.json
単位 = 騎手 / 調教師 / 騎手×調教師 / 騎手×競馬場 / 調教師×競馬場 / 騎手×芝ダ距離帯 / 調教師×芝ダ距離帯。件数の下限（単独 300、組み合わせ 100）以上の関係者について、選択年（2022〜2024 の合算）の
単勝・複勝の回収率の t 値（レース別の利益の平均 / 標準誤差）を出し、100% を超えると言えるものが偶然の数（検定数 × 有意水準）を超えるかを Benjamini-Hochberg で調べる。
その後（2025 年、2026 年）の同じ関係者の回収率も添える。2026 年はテストセット②で、選択には使わない。
"""

from __future__ import annotations

import json
from math import erf, sqrt

import numpy as np
import pandas as pd

from src.scripts.data import tp59_lib as L
from src.scripts.data import tp59_profile as P
from src.scripts.data import tp59_rules as R

SEL = (2022, 2023, 2024)
UNITS = {"j": 300, "t": 300, "jt": 100, "jv": 100, "tv": 100, "jsd": 100, "tsd": 100}


def norm_sf(t: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1 - erf(x / sqrt(2))) for x in t])


def entity_t(d: pd.DataFrame, key: str, ret: str) -> pd.DataFrame:
    """key ごとに、レース別の利益（ret − 100 × 件数）を作り、平均と標準誤差から t 値を出す。"""
    x = d[[key, "race_id", ret]].copy()
    x["n"] = 1.0
    x["p"] = x[ret] - 100.0
    rr = x.groupby([key, "race_id"], as_index=False).agg(p=("p", "sum"), n=("n", "sum"))
    g = rr.groupby(key).agg(races=("p", "size"), prof=("p", "sum"), m=("p", "mean"), s=("p", "std"), n=("n", "sum"))
    g["t"] = g["m"] / (g["s"] / np.sqrt(g["races"]))
    g["roi"] = 100 + g["prof"] / g["n"]
    return g.dropna(subset=["t"])


def bh(p: np.ndarray, q: float = 0.1) -> int:
    o = np.sort(p)
    m = len(o)
    ok = np.where(o <= q * (np.arange(1, m + 1) / m))[0]
    return int(ok.max() + 1) if len(ok) else 0


def popularity_pooled(d: pd.DataFrame) -> dict:
    """前年の出走数順位（上位 10 / 11〜30 / その他 / 新顔）の関係者の翌年の成績を年をまたいで合算（2021〜2025 と 2026）。
    超過払戻（ex）はオッズ帯の期待払戻を引いたもの（% ポイント）。人気の騎手・厩舎が買われすぎ（ex < 0）かを、レース単位ブートストラップの区間つきで見る。"""
    out: dict = {}
    for key, nm in (("jockey_id", "jockey"), ("trainer_id", "trainer")):
        parts = []
        for y in range(2021, 2027):
            prev = d[d["year"] == y - 1].groupby(key).size().sort_values(ascending=False)
            grp = pd.Series("rest", index=prev.index)
            grp.iloc[:10] = "top10"
            grp.iloc[10:30] = "11-30"
            g = d[d["year"] == y].copy()
            g["grp"] = g[key].map(grp).fillna("new")
            parts.append(g)
        a = pd.concat(parts)
        out[nm] = {}
        for per, ys in (("2021-2025", range(2021, 2026)), ("2026", (2026,))):
            x = a[a["year"].isin(list(ys))]
            rows = {}
            for k, gg in x.groupby("grp"):
                rows[k] = {"n": int(len(gg)), "win_roi": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_win"].to_numpy(), 500), "place_roi": L.boot_roi(gg["race_id"].to_numpy(), gg["ret_place"].to_numpy(), 500),
                           "ex_win": L.boot_roi(gg["race_id"].to_numpy(), gg["ex_win"].to_numpy(), 500), "ex_place": L.boot_roi(gg["race_id"].to_numpy(), gg["ex_place"].to_numpy(), 500),
                           "median_win_odds": float(gg["win_odds"].median()), "win_rate": float(gg["win"].mean()), "q_rate": float(gg["q_win"].mean())}
            out[nm][per] = rows
    return out


def run() -> dict:
    d = R.add_keys(P.load())
    sel = d[d["year"].isin(SEL)]
    fwd = {"2025": d[d["year"] == 2025], "2026": d[d["year"] == 2026]}
    out: dict = {}
    for ent, nmin in UNITS.items():
        key = f"k_{ent}"
        out[ent] = {}
        for bet in ("win", "place"):
            ret = f"ret_{bet}"
            g = entity_t(sel, key, ret)
            g = g[g["n"] >= nmin]
            if len(g) < 10:
                continue
            p = norm_sf(g["t"].to_numpy())
            top = g.sort_values("t", ascending=False).head(10)
            res = {"entities": int(len(g)), "sig_5pct_onesided": int((p < 0.05).sum()), "expected_by_chance": float(0.05 * len(g)), "bh_q10": bh(p), "t_max": float(g["t"].max()),
                   "roi_median": float(g["roi"].median()), "frac_roi_gt_100": float((g["roi"] > 100).mean()), "top": []}
            for idx, r in top.iterrows():
                row = {"key": str(idx), "n": int(r["n"]), "roi_sel": float(r["roi"]), "t": float(r["t"])}
                for nm, f in fwd.items():
                    s = f[f[key] == idx]
                    row[f"roi_{nm}"] = float(s[ret].sum() / (100.0 * len(s)) * 100) if len(s) else None
                    row[f"n_{nm}"] = int(len(s))
                res["top"].append(row)
            # 選択年の t 値の上位 10% の関係者を、その後の年に（全出走）買った回収率
            hi = g.index[g["t"] >= g["t"].quantile(0.9)]
            for nm, f in fwd.items():
                s = f[f[key].isin(hi)]
                res[f"top_decile_roi_{nm}"] = L.boot_roi(s["race_id"].to_numpy(), s[ret].to_numpy(), 500) if len(s) else None
                allf = f[f[key].isin(g.index)]
                res[f"all_roi_{nm}"] = float(allf[ret].sum() / (100.0 * len(allf)) * 100) if len(allf) else None
            # 選択年の回収率と、その後の年の回収率の順位相関（件数 50 以上の関係者）
            rho = {}
            for nm, f in fwd.items():
                h = entity_t(f, key, ret)
                h = h[h["n"] >= 50]
                com = g.index.intersection(h.index)
                rho[nm] = {"entities": int(len(com)), "spearman_roi": L.spearman(g.loc[com, "roi"].to_numpy(), h.loc[com, "roi"].to_numpy()) if len(com) > 5 else None}
            res["persistence"] = rho
            out[ent][bet] = res
    out["popularity_pooled"] = popularity_pooled(d)
    (P.OUT / "census.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


if __name__ == "__main__":
    run()
    print("saved")
