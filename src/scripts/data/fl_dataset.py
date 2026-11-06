"""最終層の入力表（馬単位・レース単位・払戻）を 1 本作る。

python -m src.scripts.data.fl_dataset build

親モデル = v6e。2022〜2024 年は LOYO の OOF（tp27_v6e/ens/v6.parquet）、2025 年は --fit-end 2024 の test 予測（final_eval_v6e/test2025_predictions.parquet）。
2026 年は使わない。オッズは確定オッズ（odds_win_place）。購入時（T-60）のオッズの代用でしかない点が最大の不確実性。
出力（FL_OUT、既定 /home/jovyan/work/fl_out）: horses.parquet / races.parquet / hits.pkl / consistency.json
除外: 払戻が曖昧・読めない・同着（単勝の払戻が 2 行以上）・1〜3 着が一意でない・3 連単の取り込み異常。
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.data.tp36_single import PAY_CAP, parse_combo  # noqa: E402

MOD = ROOT / "data/local/modeling"
BET = MOD / "betting"
OOF = MOD / "training_pipeline_v4/tp27_v6e/ens/v6.parquet"
TEST = MOD / "training_pipeline/final_eval_v6e/test2025_predictions.parquet"
TABLES = ROOT / "data/page_reference/tables"
OUT = Path(os.environ.get("FL_OUT", "/home/jovyan/work/fl_out"))
YEARS = (2022, 2023, 2024, 2025)
BTS = ("tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.concat([
        pd.read_parquet(OOF, columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"]).assign(src="oof"),
        pd.read_parquet(TEST, columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"]).assign(src="test"),
    ], ignore_index=True)
    pred["race_id"], pred["horse_id"] = pred["race_id"].astype(str), pred["horse_id"].astype(str)
    pred["year"] = pred["race_id"].str[:4].astype(int)

    flat = []
    for y in YEARS:
        f = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet",
                            columns=["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "track_condition", "field_size",
                                     "horse_id", "horse_number", "finish_position", "bracket_number"])
        flat.append(f)
    flat = pd.concat(flat, ignore_index=True)
    flat["race_id"], flat["horse_id"] = flat["race_id"].astype(str), flat["horse_id"].astype(str)
    flat = flat.drop_duplicates(["race_id", "horse_id"])

    odds = pd.read_parquet(BET / "odds_win_place.parquet", columns=["race_id", "horse_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity"])
    odds["race_id"], odds["horse_id"] = odds["race_id"].astype(str), odds["horse_id"].astype(str)
    odds = odds.drop_duplicates(["race_id", "horse_id"]).rename(columns={"horse_number": "hn_odds"})

    h = pred.merge(flat, on=["race_id", "horse_id"], how="left").merge(odds, on=["race_id", "horse_id"], how="left")
    rep: dict = {"pred_rows": int(len(pred)), "after_merge": int(len(h)), "no_flat": int(h["horse_number"].isna().sum()), "no_odds": int(h["win_odds"].isna().sum())}

    pay = pd.read_parquet(BET / "payoff.parquet")
    pay = pay[pay["race_id"].str[:4].astype(int).isin(YEARS)]
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    pay = pay[pay["bet_type"].isin(BTS)]
    hits: dict = {}
    broken: dict = {}
    for r in pay.itertuples(index=False):
        if r.race_id in bad:
            continue
        if float(r.payout) >= PAY_CAP.get(r.bet_type, float("inf")):
            broken.setdefault(r.race_id, set()).add(r.bet_type)
            continue
        hits.setdefault(r.race_id, {}).setdefault(r.bet_type, {})[parse_combo(r.combo)] = float(r.payout)
    tan_rows = pay[pay["bet_type"] == "tansho"].groupby("race_id")["n_rows"].max()
    dead = set(tan_rows[tan_rows > 1].index)

    g = h.groupby("race_id", sort=False)
    rows = []
    for rid, d in g:
        n = len(d)
        fp = d["finish_position"].to_numpy(dtype=float)
        why = ""
        if rid in bad:
            why = "payoff_ambiguous"
        elif rid in dead:
            why = "dead_heat"
        elif d["horse_number"].isna().any() or d["win_odds"].isna().any():
            why = "missing_join"
        elif not all(np.sum(fp == k) == 1 for k in (1, 2, 3)):
            why = "top3_not_unique"
        elif n < 5:
            why = "tiny_field"
        elif "tansho" not in hits.get(rid, {}) or "sanrenpuku" not in hits.get(rid, {}) or "fukusho" not in hits.get(rid, {}):
            why = "no_payoff"
        rows.append({"race_id": rid, "date": d["date"].iloc[0], "year": int(rid[:4]), "venue": d["venue"].iloc[0], "surface": d["surface"].iloc[0],
                     "distance": int(d["distance"].iloc[0]) if pd.notna(d["distance"].iloc[0]) else -1, "grade": d["grade"].iloc[0],
                     "race_class": d["race_class"].iloc[0], "track_condition": d["track_condition"].iloc[0], "n": n,
                     "src": d["src"].iloc[0], "excl": why, "broken": ",".join(sorted(broken.get(rid, ())))})
    races = pd.DataFrame(rows)
    rep["races_total"] = int(len(races))
    rep["excl_counts"] = races["excl"].replace("", "ok").value_counts().to_dict()
    ok = set(races.loc[races["excl"] == "", "race_id"])
    h = h[h["race_id"].isin(ok)].copy()
    h["horse_number"] = h["horse_number"].astype(int)
    h["finish_position"] = h["finish_position"].astype(int)
    h["fp"] = np.where(h["finish_position"] > 0, h["finish_position"], 99)  # 中止・除外などは 99
    h["month"] = pd.to_datetime(h["date"]).dt.month

    # 市場確率 = 単勝の確定オッズの逆数を正規化
    inv = 1.0 / h["win_odds"].clip(lower=1.0)
    h["q_mkt"] = inv / inv.groupby(h["race_id"]).transform("sum")
    h["pop_rank"] = h.groupby("race_id")["win_odds"].rank(method="first")
    # 単勝・複勝の払戻（100 円あたり）
    ret_w, ret_p = [], []
    for rid, hn in zip(h["race_id"], h["horse_number"]):
        hh = hits[rid]
        ret_w.append(hh["tansho"].get((hn,), 0.0))
        ret_p.append(hh["fukusho"].get((hn,), 0.0))
    h["ret_win"], h["ret_place"] = ret_w, ret_p
    h["place_mid"] = (h["place_odds_min"] + h["place_odds_max"]) / 2
    h = h.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)

    # 整合の確認
    s = h.groupby("race_id")
    rep["horses"] = int(len(h))
    rep["races_ok"] = int(h["race_id"].nunique())
    rep["races_by_year"] = h.groupby("year")["race_id"].nunique().to_dict()
    rep["p_win_sum_mean"] = float(s["p_win"].sum().mean())
    rep["p_win_sum_range"] = [float(s["p_win"].sum().min()), float(s["p_win"].sum().max())]
    rep["q_mkt_top_is_winner_rate"] = float(h.loc[h["pop_rank"] == 1, "finish_position"].eq(1).mean())
    rep["model_top_is_winner_rate_by_year"] = {int(y): float(d.loc[d.groupby("race_id")["p_win"].idxmax(), "finish_position"].eq(1).mean()) for y, d in h.groupby("year")}
    w = h[h["finish_position"] == 1]
    rep["win_odds_x100_vs_tansho_payout"] = {"abs_diff_median": float((w["win_odds"] * 100 - w["ret_win"]).abs().median()),
                                              "share_within_5pct": float(((w["win_odds"] * 100 - w["ret_win"]).abs() <= 0.05 * w["ret_win"]).mean()),
                                              "by_year_share_within_5pct": {int(y): float(((d["win_odds"] * 100 - d["ret_win"]).abs() <= 0.05 * d["ret_win"]).mean()) for y, d in w.groupby("year")}}
    rep["tansho_payback_all_buy"] = {int(y): float(d["ret_win"].sum() / (100 * len(d))) for y, d in h.groupby("year")}
    rep["fukusho_payback_all_buy"] = {int(y): float(d["ret_place"].sum() / (100 * len(d))) for y, d in h.groupby("year")}
    rep["pop1_tansho_roi_by_year"] = {int(y): float(d.loc[d["pop_rank"] == 1, "ret_win"].sum() / (100 * (d["pop_rank"] == 1).sum())) for y, d in h.groupby("year")}
    rep["sanrenpuku_hit_present_rate"] = float(np.mean([("sanrenpuku" in hits.get(r, {})) for r in ok]))
    rep["win_ll_model_by_year"] = {int(y): float(-np.log(d.loc[d["finish_position"] == 1, "p_win"].clip(lower=1e-9)).mean()) for y, d in h.groupby("year")}
    rep["win_ll_market_by_year"] = {int(y): float(-np.log(d.loc[d["finish_position"] == 1, "q_mkt"].clip(lower=1e-9)).mean()) for y, d in h.groupby("year")}

    h.to_parquet(OUT / "horses.parquet", index=False)
    races.to_parquet(OUT / "races.parquet", index=False)
    (OUT / "hits.pkl").write_bytes(pickle.dumps({r: hits[r] for r in ok}))
    (OUT / "consistency.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
