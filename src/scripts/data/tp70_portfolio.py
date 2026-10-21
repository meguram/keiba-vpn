"""tp-70 補正後の確率で推奨ポートフォリオ（BettingOptimizer）を作り直し、期待回収率の較正（cand_*）・レースの価値の段・年別の範囲を当てはめ直す。

python -m src.scripts.data.tp70_portfolio build --preds tp70_cal_preds.parquet --bet DIR --bet26 DIR --mode raw|pc --years 2023 2024 2025 2026 [--shard i n] [--limit N] --out DIR
python -m src.scripts.data.tp70_portfolio fit   --out DIR [--config config/ev_calibration_pc.json]      # build の出力から cand_*・levels・range を作り、PC 設定に追記

build: calibrate_ev build_portfolio と同じ手順（max_total_bet=2000・単勝/複勝/馬連/ワイド・15 点）を、同じレース・同じオッズ・同じ払戻で、
  raw = モデルの確率 / pc = 市場補正した確率（前年までで当てはめた walk-forward。勝率=C3・連対/複勝=C6）で回す。
fit: 方式選択は 2024・2025 年（前年までで当てはめ）、2026 年は最終確認だけ。設定の係数は 2023〜2026 年で当てはめた目安（従来と同じ扱い）。
市場の確率はモデルの特徴量にはしない（較正と馬券判断だけ）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import audit_ev_calibration as au
from src.scripts.data import calibrate_ev as ce

PC_COLS = {"p_win": "cal_C3_win", "p_top2": "cal_C6_top2", "p_top3": "cal_C6_top3"}
CAND = ("tansho", "fukusho", "umaren", "wide")


def load_inputs(preds: str, bet: str, bet26: str, years: list[int]):
    d = pd.read_parquet(preds)
    d["race_id"] = d["race_id"].astype(str)
    d = d[d["year"].isin(years)].copy()
    d = d.dropna(subset=list(PC_COLS.values()))
    w26 = pd.read_parquet(Path(bet26) / "odds_win_place.parquet")[["race_id", "horse_number", "place_odds_min", "place_odds_max"]]
    w26["race_id"] = w26["race_id"].astype(str)
    m26 = d["year"] == 2026
    if m26.any():
        x = d[m26].drop(columns=["place_odds_min", "place_odds_max"]).merge(w26, on=["race_id", "horse_number"], how="left")
        d = pd.concat([d[~m26], x], ignore_index=True)
    d = d[d["place_odds_min"].notna() & d["place_odds_max"].notna() & (d["place_odds_min"] > 0)]
    pay = pd.concat([pd.read_parquet(Path(bet) / "payoff.parquet"), pd.read_parquet(Path(bet26) / "payoff.parquet")], ignore_index=True)
    if {"ambiguous", "parsed"} <= set(pay.columns):
        bad = set(pay.loc[pay["ambiguous"].fillna(False) | ~pay["parsed"].fillna(True), "race_id"])
        pay = pay[~pay["race_id"].isin(bad)]
    pairs = pd.concat([pd.read_parquet(Path(bet) / "odds_pair.parquet"), pd.read_parquet(Path(bet26) / "odds_pair.parquet")], ignore_index=True)
    pairs["race_id"] = pairs["race_id"].astype(str)
    pay["race_id"] = pay["race_id"].astype(str)
    ok = set(d["race_id"]) & set(pairs["race_id"]) & set(pay["race_id"])
    # 取りこぼしのあるレース（一部の馬の確率が無い）は除く
    cnt = d.groupby("race_id").size()
    return d[d["race_id"].isin(ok)], pay[pay["race_id"].isin(ok)], pairs[pairs["race_id"].isin(ok)]


def build(a) -> None:
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer

    d, pay, pairs = load_inputs(a.preds, a.bet, a.bet26, a.years)
    if a.mode == "pc":
        d = d.assign(**{k: d[v] for k, v in PC_COLS.items()})
    hit = au._payoff_dicts(pay)
    pair_by_race = {r: g for r, g in pairs.groupby("race_id")}
    opt = BettingOptimizer(BettingConfig(max_total_bet=2000, bet_types=list(CAND)))
    ids = sorted(d["race_id"].unique())
    if a.shard:
        ids = ids[a.shard[0] :: a.shard[1]]
    if a.limit:
        ids = ids[: a.limit]
    dg = {r: g for r, g in d.groupby("race_id")}
    races, cand = [], []
    for n, r in enumerate(ids):
        g = dg[r]
        ent, odds = au._entries_for(g)
        preds = [{**e, "win_odds": odds[e["horse_number"]]["win_odds"], "place_odds_min": odds[e["horse_number"]]["place_odds_min"],
                  "place_odds_max": odds[e["horse_number"]]["place_odds_max"], "pred_score": math.log(max(e["win_prob"], 1e-9))} for e in ent]
        single = {"entries": [{"horse_number": hn, **o} for hn, o in odds.items()]}
        port = opt.optimize(preds, au._pair_odds_docs(pair_by_race[r]), 100000, single_odds=single)
        cs = port["candidates"]
        if not cs:
            races.append({"race_id": r, "year": int(r[:4]), "n_bets": 0, "claimed": np.nan, "realized": np.nan, "stake": 0, "ret": 0.0})
            continue
        stake = float(sum(c.bet_amount for c in cs))
        ret = 0.0
        for c in cs:
            key = str(c.pair[0]) if c.bet_type in ("tansho", "fukusho") else au._key2(*c.pair)
            u = hit[c.bet_type].get(r, {}).get(key, 0.0) / 100
            ret += u * c.bet_amount
            cand.append({"race_id": r, "year": int(r[:4]), "bet_type": c.bet_type, "ev": float(c.ev), "odds": float(c.odds), "prob": float(c.prob), "ret": u, "amount": float(c.bet_amount)})
        races.append({"race_id": r, "year": int(r[:4]), "n_bets": len(cs), "claimed": float(port["expected_roi"]), "realized": ret / stake, "stake": stake, "ret": ret})
        if (n + 1) % 200 == 0:
            print(a.mode, n + 1, "/", len(ids), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.mode}_{'_'.join(map(str, a.years))}" + (f"_s{a.shard[0]}of{a.shard[1]}" if a.shard else "")
    pd.DataFrame(races).to_parquet(out / f"pf_race_{tag}.parquet", index=False)
    pd.DataFrame(cand).to_parquet(out / f"pf_cand_{tag}.parquet", index=False)
    print("保存:", tag, len(races), len(cand))


def _load(out: Path, mode: str):
    c = pd.concat([pd.read_parquet(p) for p in sorted(out.glob(f"pf_cand_{mode}_*.parquet"))], ignore_index=True)
    r = pd.concat([pd.read_parquet(p) for p in sorted(out.glob(f"pf_race_{mode}_*.parquet"))], ignore_index=True)
    return c, r


def race_pred(c: pd.DataFrame, methods: dict[str, str], train_end: int | None, years: list[int], cap: float) -> pd.DataFrame:
    """候補の EV を較正（前年までで当てはめ）し、レースの期待回収率（購入額加重）にする。"""
    rows = []
    for y in years:
        tr = c[c["year"] < y] if train_end is None else c[c["year"] <= train_end]
        te = c[c["year"] == y].copy()
        te["cal"] = ce._cal_by_kind(tr, te, methods)
        te = te.dropna(subset=["cal"])
        te["w"] = te["cal"] * te["amount"]
        g = te.groupby("race_id").agg(w=("w", "sum"), a=("amount", "sum")).reset_index()
        g["pred"] = (g["w"] / g["a"]).clip(upper=cap)
        g["year"] = y
        rows.append(g)
    return pd.concat(rows)


def fit(a) -> None:
    out = Path(a.out)
    cap = 1.15
    res: dict = {}
    pc_c, pc_r = _load(out, "pc")
    raw_c, raw_r = _load(out, "raw")
    # 年別の実現（購入額加重）と、従来の主張（claimed）との差
    def yearly(r: pd.DataFrame) -> dict:
        r = r[r["n_bets"] > 0]
        return {int(y): {"races": int(len(g)), "realized": round(float(g["ret"].sum() / g["stake"].sum()), 4), "claimed": round(float(np.average(g["claimed"], weights=g["stake"])), 4), "stake": float(g["stake"].sum())} for y, g in r.groupby("year")}
    res["yearly"] = {"raw": yearly(raw_r), "pc": yearly(pc_r)}
    # cand_* の方式選択: 2024・2025 年を、前年までで当てはめて評価（pc の候補）
    sel = {}
    for bt in CAND:
        g = pc_c[pc_c["bet_type"] == bt]
        by = ce.walk_forward(g, "ev", "ret", ce.BASE[ce.KIND_OF_BET.get(bt, bt)], "amount", years=(2024, 2025), methods=ce.METHODS)
        sel[bt] = ce.choose_method(by)
    methods = {bt: (sel[bt][0] or "const") for bt in CAND}
    res["methods"] = methods
    res["selection"] = {bt: {"method": sel[bt][0], "ok": bool(sel[bt][1].get("ok")), "pooled": sel[bt][1].get("pooled")} for bt in CAND}
    # 2024・2025 年（前年まで）と 2026 年（2023〜2025 年で当てはめ）の、レース単位の期待回収率の実現との対応
    comp = {}
    for label, years, train_end in (("2024-25", [2024, 2025], None), ("2026", [2026], 2025)):
        P = race_pred(pc_c, methods, train_end, years, cap).merge(pc_r[["race_id", "realized", "stake", "ret", "claimed"]], on="race_id")
        comp[label] = {"calibrated": {k: v for k, v in ce.calib_metrics(P["pred"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()).items() if k != "bins"},
                       "claimed_raw": {k: v for k, v in ce.calib_metrics(P["claimed"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()).items() if k != "bins"}}
        comp[label]["pooled_realized"] = round(float(P["ret"].sum() / P["stake"].sum()), 4)
    res["race_compare"] = comp
    # 従来（補正しない確率）のポートフォリオの同じ指標
    comp_raw = {}
    sel_raw = {}
    for bt in CAND:
        g = raw_c[raw_c["bet_type"] == bt]
        by = ce.walk_forward(g, "ev", "ret", ce.BASE[ce.KIND_OF_BET.get(bt, bt)], "amount", years=(2024, 2025), methods=ce.METHODS)
        sel_raw[bt] = ce.choose_method(by)
    m_raw = {bt: (sel_raw[bt][0] or "const") for bt in CAND}
    for label, years, train_end in (("2024-25", [2024, 2025], None), ("2026", [2026], 2025)):
        P = race_pred(raw_c, m_raw, train_end, years, cap).merge(raw_r[["race_id", "realized", "stake", "ret", "claimed"]], on="race_id")
        comp_raw[label] = {"calibrated": {k: v for k, v in ce.calib_metrics(P["pred"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()).items() if k != "bins"},
                           "claimed_raw": {k: v for k, v in ce.calib_metrics(P["claimed"].to_numpy(), P["realized"].to_numpy(), P["race_id"].to_numpy(), P["stake"].to_numpy()).items() if k != "bins"}}
    res["race_compare_raw_portfolio"] = comp_raw
    # 設定の cand_*（2023〜2026 年で当てはめ）と、レース価値の段
    kinds = {}
    for bt in CAND:
        g = pc_c[pc_c["bet_type"] == bt]
        spec = ce.fit_method(methods[bt], g["ev"].to_numpy(), g["ret"].to_numpy(), ce.BASE[ce.KIND_OF_BET.get(bt, bt)], g["amount"].to_numpy())
        c26 = comp["2026"]["calibrated"]
        kinds[f"cand_{bt}"] = {**spec, "validated": bool(sel[bt][1].get("ok")), "n": int(len(g)), "note": "補正後の確率のポートフォリオで 2023〜2026 年に当てはめた目安（方式は 2024・25 年で前年までの当てはめで選択）"}
    for bt in CAND:                           # 標本が少なく検証も通らない定数は信頼できない（複勝: 2026 年は複勝オッズが無く 2023〜25 年の少数の買い目だけ）→ 市場並みの払戻率に置く
        k = kinds[f"cand_{bt}"]
        if k["method"] == "const" and not k["validated"] and k["n"] < 2000:
            kinds[f"cand_{bt}"] = {**ce.conservative(ce.BASE[ce.KIND_OF_BET.get(bt, bt)]), "validated": False, "n": k["n"],
                                   "note": f"買い目 {k['n']} 点と少なく、検証も通らないため定数（{k['value']:.3f}）は使わず市場並みの払戻率に置いた"}
    c_all = pc_c.copy()
    c_all["cal"] = np.nan
    for bt in CAND:
        idx = c_all["bet_type"] == bt
        c_all.loc[idx, "cal"] = ce.apply_spec(kinds[f"cand_{bt}"], c_all.loc[idx, "ev"].to_numpy())
    cw = c_all.assign(w=c_all["cal"] * c_all["amount"]).groupby("race_id").agg(w=("w", "sum"), a=("amount", "sum")).reset_index()
    cw["pred"] = (cw["w"] / cw["a"]).clip(upper=cap)
    P = pc_r[pc_r["n_bets"] > 0].merge(cw[["race_id", "pred"]], on="race_id")
    old = json.loads(Path(a.config_standard).read_text(encoding="utf-8")) if a.config_standard else {}
    old_lv = old.get("levels") or {"hot": 1.03, "good": 1.0, "ok": 0.97}
    res["pred_quantiles"] = {str(q): round(float(P["pred"].quantile(q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    # 従来の段の“頭の割合”を保つ段: 従来の設定の閾値でのレースの割合（従来のポートフォリオ・従来の較正）に合わせて、補正後の分布の分位点から閾値を決める
    Pr = raw_r[raw_r["n_bets"] > 0].merge(race_pred(raw_c, m_raw, None, [2023, 2024, 2025, 2026], cap)[["race_id", "pred"]], on="race_id")
    shares = {k: float((Pr["pred"] >= old_lv[k]).mean()) for k in ("hot", "good", "ok")}
    new_lv = {k: math.floor(float(P["pred"].quantile(1 - shares[k])) * 1000) / 1000 for k in ("hot", "good", "ok")}      # 離散的な値なので切り捨てて、その値以上が閾値になるようにする
    res["level_shares_old"] = {k: round(v, 4) for k, v in shares.items()}
    res["levels_old"], res["levels_new_share_matched"] = old_lv, new_lv
    grid = sorted(set([0.90, 0.93, 0.95] + list(new_lv.values()) + list(old_lv.values())))
    res["levels_table_pc"] = ce._level_table(P["pred"].to_numpy(), P["ret"].to_numpy(), P["stake"].to_numpy(), P["race_id"].to_numpy(), grid)
    res["levels_table_pc_by_year"] = {str(y): ce._level_table(g["pred"].to_numpy(), g["ret"].to_numpy(), g["stake"].to_numpy(), g["race_id"].to_numpy(), grid) for y, g in P.groupby("year")}
    # 混在の確認: 従来の設定（従来のポートフォリオ・従来の較正）と、補正後の設定（補正後のポートフォリオ・PC 較正）で、同じ段の割合と実現
    res["mix_check"] = {"old_levels_on_raw": {k: {"share": round(float((Pr['pred'] >= old_lv[k]).mean()), 4), "real": round(float(Pr.loc[Pr['pred'] >= old_lv[k], 'ret'].sum() / max(Pr.loc[Pr['pred'] >= old_lv[k], 'stake'].sum(), 1)), 4)} for k in old_lv},
                        "new_levels_on_pc": {k: {"share": round(float((P['pred'] >= new_lv[k]).mean()), 4), "real": round(float(P.loc[P['pred'] >= new_lv[k], 'ret'].sum() / max(P.loc[P['pred'] >= new_lv[k], 'stake'].sum(), 1)), 4)} for k in new_lv},
                        "old_levels_on_pc": {k: {"share": round(float((P['pred'] >= old_lv[k]).mean()), 4), "real": round(float(P.loc[P['pred'] >= old_lv[k], 'ret'].sum() / max(P.loc[P['pred'] >= old_lv[k], 'stake'].sum(), 1)), 4)} for k in old_lv}}
    yearly_pc = res["yearly"]["pc"]
    lo = min(v["realized"] for v in yearly_pc.values())
    hi = max(v["realized"] for v in yearly_pc.values())
    res["config_patch"] = {"kinds": kinds, "levels": new_lv, "range": {"lo": lo, "hi": hi, "basis": "2023〜2026 年の年別の実現回収率（購入額加重。補正後の確率のポートフォリオ）", "yearly": {str(y): v["realized"] for y, v in yearly_pc.items()}}}
    (out / "tp70_portfolio.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    if a.config:
        p = Path(a.config)
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg["kinds"].update(kinds)
        cfg["levels"] = {**new_lv, "hot": 9.99} if a.hot_off else new_lv
        cfg["levels_share_matched"] = new_lv
        cfg["range"] = res["config_patch"]["range"]
        cfg["levels_note"] = ("補正後の確率のポートフォリオを 2023〜2026 年で作り直した（tp-70）。範囲・cand_*・段はその再計算。"
                              + ("補正後はレースごとの期待回収率が券種構成（定数の較正）でほぼ決まり、予測の高い順に実現が上がらない（五分位の実現が非単調、hot 相当は全てワイドのレースで実現 89%）ため、"
                                 "hot（★オレンジ）は出さない（閾値 9.99）。good/ok は従来の頭の割合に合わせた。hot を戻すなら levels.hot = levels_share_matched.hot。" if a.hot_off else "段は従来の段の頭の割合に合わせた。"))
        cfg["fit"] = {**(cfg.get("fit") or {}), "pc_portfolio": "2023-2026（補正後の確率。確定オッズ。max_total_bet=2000, 15 点, 単勝・複勝・馬連・ワイド）"}
        cfg["note"] = cfg.get("note", "").replace("（連対 EV の実現が無い／ポートフォリオの再実行が必要。calibrate_ev build_portfolio を補正後の確率で再実行して差し替える）", "（cand_*・levels・range は tp70_portfolio で補正後の確率のポートフォリオを 2023〜2026 年に再実行して作った）")
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("yearly", "methods", "race_compare", "levels_old", "levels_new_share_matched", "level_shares_old", "mix_check")}, ensure_ascii=False, indent=1, default=lambda o: None)[:6000])


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--preds", required=True)
    p.add_argument("--bet", required=True)
    p.add_argument("--bet26", required=True)
    p.add_argument("--mode", choices=["raw", "pc"], required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    p.add_argument("--shard", type=int, nargs=2, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=build)
    p = sp.add_parser("fit")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--config-standard", default="config/ev_calibration.json")
    p.add_argument("--hot-off", action="store_true", help="補正後は hot（★オレンジ）を出さない（予測の高い順に実現が上がらないため）")
    p.set_defaults(fn=fit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
