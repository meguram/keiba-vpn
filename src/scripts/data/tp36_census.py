"""tp-36 (1) 構造的な歪みの全数調査: 券種×人気（オッズ帯）×条件ごとの『均等買いの回収率』を年別・信頼区間つきで出す（モデル不使用）。

python -m src.scripts.data.tp36_census --out data/local/modeling/betting/tp36/census.json
入力: data/local/modeling/betting/{odds_win_place,odds_pair,payoff}.parquet（2020〜2025 年の確定オッズと払戻。曖昧なレースは除く）。
回収率 = 払戻の合計 / 賭け金の合計（1 点 100 円）。信頼区間はレース単位のブートストラップ（95%）。
注意: 2020〜2025 年のオッズは締切後の確定オッズで、購入時点の値ではない。
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TRIPLE_LAST_YEAR = 2023
PAYBACK = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    w = pd.read_parquet(BET / "odds_win_place.parquet")
    pair = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "popularity"])
    pay = pd.read_parquet(BET / "payoff.parquet")
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    good = set(pay["race_id"]) - bad
    w = w[w["race_id"].isin(good)].copy()
    pair = pair[pair["race_id"].isin(good)].copy()
    pay = pay[pay["race_id"].isin(good)].copy()
    for d in (w, pair, pay):
        d["year"] = d["race_id"].str[:4].astype(int)
    return w, pair, pay


def boot_roi(g: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float, int]:
    """g: race_id, stake, ret（レース単位に集計済み）。戻り値: roi, lo, hi, 賭け数。"""
    s, r = g["stake"].to_numpy(float), g["ret"].to_numpy(float)
    if s.sum() <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    roi = r.sum() / s.sum()
    rng = np.random.default_rng(seed)
    n = len(s)
    idx = rng.integers(0, n, size=(n_boot, n))
    ss, rr = s[idx].sum(1), r[idx].sum(1)
    ok = ss > 0
    b = rr[ok] / ss[ok]
    return float(roi), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), int(round(s.sum() / 100))


def band_label(lo: int, hi: int | None) -> str:
    return f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")


def table_by_band(per: pd.DataFrame, bands: list[tuple[int, int | None]], key: str) -> list[dict]:
    """per: race_id, year, <key>=人気順位, stake_unit(1 点なら 100), ret。key の値で帯に分ける。"""
    rows = []
    for lo, hi in bands:
        m = (per[key] >= lo) & ((per[key] <= hi) if hi is not None else True)
        sub = per[m]
        agg = sub.groupby(["race_id", "year"], sort=False).agg(stake=("stake", "sum"), ret=("ret", "sum")).reset_index()
        pooled = boot_roi(agg)
        row = {"band": band_label(lo, hi), "roi": pooled[0], "lo": pooled[1], "hi": pooled[2], "n_bets": pooled[3]}
        for y in range(2020, 2026):
            a = agg[agg["year"] == y]
            row[str(y)] = float(a["ret"].sum() / a["stake"].sum()) if a["stake"].sum() > 0 else None
        rows.append(row)
    return rows


def census_horse_bets(w: pd.DataFrame, pay: pd.DataFrame, bt: str) -> pd.DataFrame:
    p = pay[pay["bet_type"] == bt][["race_id", "combo", "payout"]].copy()
    p["horse_number"] = p["combo"].astype(int)
    d = w[["race_id", "year", "horse_number", "popularity", "win_odds"]].merge(p[["race_id", "horse_number", "payout"]], on=["race_id", "horse_number"], how="left")
    d["ret"] = d["payout"].fillna(0.0)
    d["stake"] = 100.0
    return d


def census_pairs(w: pd.DataFrame, pair: pd.DataFrame, pay: pd.DataFrame, bt: str) -> pd.DataFrame:
    """組合せの人気順位ごとの賭け数と払戻（1 点 100 円）。賭け数はレースの全組合せ。"""
    pr = pair[pair["bet_type"] == bt][["race_id", "year", "combo", "popularity"]].copy()
    p = pay[pay["bet_type"] == bt][["race_id", "combo", "payout"]]
    pr = pr.merge(p, on=["race_id", "combo"], how="left")
    pr["ret"] = pr["payout"].fillna(0.0)
    pr["stake"] = 100.0
    return pr


def census_triple(w: pd.DataFrame, pay: pd.DataFrame, bt: str, bands: list[tuple[int, int | None]]) -> list[dict]:
    """3 連複・3 連単: 全組合せのオッズ表が無いので、当たりの組合せの人気順位と払戻、全組合せ数から帯別の回収率を出す。"""
    w = w[w["year"] <= TRIPLE_LAST_YEAR]      # 払戻の人気順位は 2024 年以降が欠損のため、当たり組合せの人気で帯分けできる年に限る
    n = w.groupby("race_id").size().rename("n_field").reset_index()
    p = pay[pay["bet_type"] == bt][["race_id", "year", "payout", "popularity"]].merge(n, on="race_id", how="left")
    n_all = n.merge(w[["race_id", "year"]].drop_duplicates("race_id"), on="race_id")
    n_all["n_comb"] = n_all["n_field"].map(lambda k: comb(int(k), 3) if bt == "sanrenpuku" else int(k) * (int(k) - 1) * (int(k) - 2))
    rows = []
    for lo, hi in bands:
        a = n_all.copy()
        top = a["n_comb"] if hi is None else a["n_comb"].clip(upper=hi)
        a["stake"] = 100.0 * (top - lo + 1).clip(lower=0)
        h = p.copy()
        m = (h["popularity"] >= lo) & ((h["popularity"] <= hi) if hi is not None else True)
        h["ret"] = np.where(m, h["payout"], 0.0)
        g = a[["race_id", "year", "stake"]].merge(h.groupby("race_id")["ret"].sum().reset_index(), on="race_id", how="left").fillna({"ret": 0.0})
        pooled = boot_roi(g)
        row = {"band": band_label(lo, hi), "roi": pooled[0], "lo": pooled[1], "hi": pooled[2], "n_bets": pooled[3]}
        for y in range(2020, TRIPLE_LAST_YEAR + 1):
            gy = g[g["year"] == y]
            row[str(y)] = float(gy["ret"].sum() / gy["stake"].sum()) if gy["stake"].sum() > 0 else None
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BET / "tp36/census.json"))
    a = ap.parse_args()
    w, pair, pay = load()
    out: dict = {"races": int(w["race_id"].nunique()), "years": sorted(w["year"].unique().tolist()), "payback": PAYBACK}
    pop_bands = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 12), (13, None)]
    for bt in ("tansho", "fukusho"):
        d = census_horse_bets(w, pay, bt)
        out[bt] = {"by_popularity": table_by_band(d, pop_bands, "popularity")}
        # オッズ帯（単勝オッズで帯分け）
        ob = [(0, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 10_000)]
        rows = []
        for lo, hi in ob:
            sub = d[(d["win_odds"] >= lo) & (d["win_odds"] < hi)]
            agg = sub.groupby(["race_id", "year"]).agg(stake=("stake", "sum"), ret=("ret", "sum")).reset_index()
            pooled = boot_roi(agg)
            row = {"band": f"{lo}-{hi}", "roi": pooled[0], "lo": pooled[1], "hi": pooled[2], "n_bets": pooled[3]}
            for y in range(2020, 2026):
                ay = agg[agg["year"] == y]
                row[str(y)] = float(ay["ret"].sum() / ay["stake"].sum()) if ay["stake"].sum() > 0 else None
            rows.append(row)
        out[bt]["by_odds"] = rows
        print(bt, "done", flush=True)
    for bt in ("umaren", "wide", "umatan"):
        pr = census_pairs(w, pair, pay, bt)
        bands = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 40), (41, 80), (81, None)]
        out[bt] = {"by_popularity": table_by_band(pr, bands, "popularity")}
        print(bt, "done", flush=True)
    for bt in ("sanrenpuku", "sanrentan"):
        bands = [(1, 1), (2, 5), (6, 20), (21, 60), (61, 150), (151, 400), (401, 1000), (1001, None)]
        out[bt] = {"by_popularity": census_triple(w, pay, bt, bands)}
        print(bt, "done", flush=True)
    # 条件別（単勝・複勝: 1 番人気と人気薄側）
    cond = w[["race_id"]].drop_duplicates()
    base = pd.read_parquet(ROOT / "data/page_reference/tables/2025/race_result_flat.parquet", columns=["race_id"]).head(0)
    outc = {}
    try:
        meta = pd.concat([pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet", columns=["race_id", "surface", "distance", "grade", "venue_code", "field_size", "track_condition"]).drop_duplicates("race_id") for y in range(2020, 2026)])
        meta["dist_band"] = pd.cut(meta["distance"].astype(float), [0, 1300, 1700, 2100, 2500, 9999], labels=["~1300", "1400-1700", "1800-2100", "2200-2500", "2600~"]).astype(str)
        meta["field_band"] = pd.cut(meta["field_size"].astype(float), [0, 11, 14, 16, 99], labels=["~11", "12-14", "15-16", "17-"]).astype(str)
        for bt in ("tansho", "fukusho"):
            d = census_horse_bets(w, pay, bt).merge(meta, on="race_id", how="left")
            res = []
            for col in ("surface", "dist_band", "field_band", "grade", "venue_code", "track_condition"):
                for val, g in d.groupby(col):
                    for tag, m in (("人気1-3", g["popularity"] <= 3), ("人気9+", g["popularity"] >= 9)):
                        sub = g[m]
                        if len(sub) < 3000:
                            continue
                        agg = sub.groupby(["race_id", "year"]).agg(stake=("stake", "sum"), ret=("ret", "sum")).reset_index()
                        roi, lo, hi, nb = boot_roi(agg, n_boot=400)
                        ys = {y: float(a2["ret"].sum() / a2["stake"].sum()) for y, a2 in agg.groupby("year") if a2["stake"].sum() > 0}
                        res.append({"cond": col, "value": str(val), "pop": tag, "roi": roi, "lo": lo, "hi": hi, "n_bets": nb, "min_year": min(ys.values()), "max_year": max(ys.values())})
            outc[bt] = sorted(res, key=lambda r: -r["roi"])[:25]
    except Exception as e:  # noqa: BLE001
        outc["error"] = str(e)
    out["conditions_top"] = outc
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
