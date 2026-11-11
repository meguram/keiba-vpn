"""想定オッズ用の『過去走の市場指標』（その日より前に確定した値だけ）を as-of で作る。ホストで実行。

    REBUILD_2016=1 python -m src.scripts.data.odds_prev_market_feat --out retrain_out/odds_prev_mk/feat.parquet

作る列（列名は odds・人気を含まない `pa_mk_*`。許可は pred_odds.ALLOW_PREV_MARKET で明示）:
- 馬の直近 1〜3 走の log 単勝（lo1〜lo3）、直近 3/5 走の平均・直近 5 走の最小/最大
- 直近 1/3 走の人気順位の比（rkr: 0=1番人気 〜 1=最低人気）と、人気より着順が良かったか（bt = 人気比 − 着順比。正は人気以上）
- 直近 5 走の 2 番人気以内の回数、前走レースの 1 番人気の log 単勝（相手の強さ）と前走の自分の log 単勝との差
- 騎手・調教師の、それより前の日までの全走の平均 log 単勝・人気順位の比（縮め先は全体平均、k=30）
値は同じ馬の前の走（日付が厳密に前）と、キーごとに『前の日まで』の累積だけから作る。当日の市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import odds_pred_eval as oe

KEY = ["race_id", "horse_id"]
SHRINK_K = 30.0


def _log(m: str) -> None:
    print(m, flush=True)


def load_runs(train_table: Path, wp: pd.DataFrame) -> pd.DataFrame:
    tt = pd.read_parquet(train_table, columns=KEY + ["date", "jockey_id", "trainer_id", "finish_position"])
    tt["race_id"], tt["horse_id"] = tt["race_id"].astype(str), tt["horse_id"].astype(str)
    tt["date"] = pd.to_datetime(tt["date"])
    d = tt.merge(wp[KEY + ["win_odds", "popularity"]], on=KEY, how="left")
    n = d.groupby("race_id")["win_odds"].transform("count")
    d["lo"] = np.log(d["win_odds"])
    pop = d["popularity"].where(d["popularity"] > 0)
    den = (n - 1).clip(lower=1)
    d["rkr"] = (pop - 1) / den
    fin = d["finish_position"].where(d["finish_position"] > 0)
    d["bt"] = d["rkr"] - (fin - 1) / den
    d["fav2"] = (pop <= 2).astype(float).where(pop.notna())
    d["race_fav_lo"] = d.groupby("race_id")["lo"].transform("min")
    return d.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)


def horse_features(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("horse_id", sort=False)
    out = pd.DataFrame({"race_id": d["race_id"], "horse_id": d["horse_id"]})
    sh = {c: [g[c].shift(k) for k in range(1, 6)] for c in ("lo", "rkr", "bt", "fav2", "race_fav_lo")}
    for k in range(3):
        out[f"pa_mk_lo{k + 1}"] = sh["lo"][k]
    L = pd.concat(sh["lo"], axis=1)
    out["pa_mk_lo_m3"] = L.iloc[:, :3].mean(axis=1)
    out["pa_mk_lo_m5"] = L.mean(axis=1)
    out["pa_mk_lo_min5"] = L.min(axis=1)
    out["pa_mk_lo_max5"] = L.max(axis=1)
    out["pa_mk_rkr1"] = sh["rkr"][0]
    out["pa_mk_rkr_m3"] = pd.concat(sh["rkr"][:3], axis=1).mean(axis=1)
    out["pa_mk_bt1"] = sh["bt"][0]
    out["pa_mk_bt_m3"] = pd.concat(sh["bt"][:3], axis=1).mean(axis=1)
    out["pa_mk_fav2_n5"] = pd.concat(sh["fav2"], axis=1).sum(axis=1, min_count=1)
    out["pa_mk_prevfav_lo"] = sh["race_fav_lo"][0]
    out["pa_mk_lo1_gap"] = sh["lo"][0] - sh["race_fav_lo"][0]
    return out


def key_features(d: pd.DataFrame, key: str, tag: str) -> pd.DataFrame:
    """key の『前の日まで』の全走の平均（縮め先は全体平均）。同じ日の走は含めない。"""
    res = d[["race_id", "horse_id", "date", key]].copy()
    for c, name in (("lo", "lo"), ("rkr", "rkr")):
        s = d[c]
        t = pd.DataFrame({key: d[key], "date": d["date"], "s": s.fillna(0.0), "n": s.notna().astype(float)})
        day = t.groupby([key, "date"], sort=True)[["s", "n"]].sum()
        cum = day.groupby(level=0).cumsum().groupby(level=0).shift(1)
        m = float(s.mean())
        mean = ((cum["s"].fillna(0.0) + SHRINK_K * m) / (cum["n"].fillna(0.0) + SHRINK_K)).rename(f"pa_mk_{tag}_{name}")
        res = res.merge(mean.reset_index(), on=[key, "date"], how="left")
    return res.drop(columns=["date", key])


def build(train_table: Path, wp: pd.DataFrame) -> pd.DataFrame:
    d = load_runs(train_table, wp)
    f = horse_features(d)
    for key, tag in (("jockey_id", "jk"), ("trainer_id", "tr")):
        f = f.merge(key_features(d, key, tag), on=KEY, how="left")
    chk = d.groupby("horse_id")["date"].apply(lambda s: s.is_monotonic_increasing).all()
    assert chk, "馬ごとに日付順に並んでいない"
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-table", default=None)
    a = ap.parse_args()
    if os.environ.get("REBUILD_2016") == "1":
        from src.scripts.data import rebuild_groups_2016 as rg

        rg.redirect()
        oe.BET = rg.BASE / "betting"
        default_tt = rg.BASE / "features/train_table.parquet"
    else:
        default_tt = Path("data/local/features/train_table.parquet")
    wp = oe.load_wp()
    f = build(Path(a.train_table) if a.train_table else default_tt, wp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(a.out, index=False)
    _log(f"保存 {a.out}: {len(f):,} 行 × {f.shape[1] - 2} 列、欠損率 " + ", ".join(f"{c}={f[c].isna().mean():.2f}" for c in f.columns[2:8]))


if __name__ == "__main__":
    main()
