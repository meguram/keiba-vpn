"""今の推奨馬券（target_bets の既定プラン）の印のずれ（betting_patterns.md §4 の W4）と R1〜R8 の違反の率を測る（読み取りのみ）。

python -m src.scripts.data.bet_cycles_w4 [--period train] [--threads 4] [--limit N]

- 今の推奨は、期待値で買い目を作り、bet_marks.aligned_mark_types で「買い目から」印を決め直す。
  そのため表示の上では印と買い目は矛盾しないが、モデルの ◎（勝率が最大の馬）とはずれうる（W4）。
- ここでは、予測の OOF（勝率・連対率・複勝率）と確定の単複オッズ（odds_mode=actual 相当。複勝は下限・上限）を入れて
  build_target_portfolio（上限 2,000 円）を手元で動かし、次を数える。
  - 買い目のあるレースのうち、買い目から決めた ◎ がモデルの ◎ と違う割合（W4）
  - モデルの印（勝率の順に ◎○▲△）で見たときの R2〜R8 の違反の割合
  - 参考: その推奨の必須の指標（確定払戻で精算）
- 馬連・ワイド・馬単・三連系の想定オッズは本番と同じく予測の構成（odds_pred の poly2）。確定オッズではない。
- 係数（config/odds_pred/odds_pred_config.json）は公開リポジトリに置かない配布物。手元では学習の出力を
  KEIBA_ODDS_PRED_CONFIG=data/local/modeling/odds_pred/odds_pred_config.json で指す（無いと全レースが見送りになる）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import bet_cycles as bc

ROOT = bc.ROOT
ODDS_WP = ROOT / "data/local/modeling/betting/odds_win_place.parquet"


def _race_inputs(r: bc.Race, p2: np.ndarray, wp: pd.DataFrame | None) -> tuple[list[dict], dict | None]:
    ents = [{"horse_number": int(h), "win_prob": float(w), "top2_prob": float(t2), "top3_prob": float(t3), "mark_type": ""}
            for h, w, t2, t3 in zip(r.hn, r.pw, p2, r.p3)]
    single = None
    if wp is not None and len(wp):
        by = wp.set_index("horse_number")
        se = []
        for h, o in zip(r.hn, r.odds):
            lo = by["place_odds_min"].get(h) if h in by.index else None
            hi = by["place_odds_max"].get(h) if h in by.index else None
            se.append({"horse_number": int(h), "win_odds": float(o) if np.isfinite(o) else None,
                       "place_odds_min": float(lo) if lo is not None and np.isfinite(lo) else None,
                       "place_odds_max": float(hi) if hi is not None and np.isfinite(hi) else None})
        single = {"entries": se, "odds_model": "actual"}
    return ents, single


_G: dict = {}


def _one(i: int) -> dict:
    from src.pipeline.inference import bet_marks as BM
    from src.pipeline.inference import target_bets as TB

    r, p2, wp = _G["races"][i], _G["p2"][i], _G["wp"].get(_G["races"][i].race_id)
    ents, single = _race_inputs(r, p2, wp)
    try:
        out = TB.build_target_portfolio(ents, single, 2000, race_id=r.race_id)
    except Exception as e:  # noqa: BLE001
        return {"race_id": r.race_id, "status": "error", "err": str(e)[:80]}
    bets = out.get("bets") if out.get("status") == "bet" else []
    rec = {"race_id": r.race_id, "status": out.get("status"), "cost": int(sum(b["yen"] for b in bets)), "ret": 0, "w4": None, "viol": ""}
    if not bets:
        return rec
    rec["ret"] = int(sum(r.pay[b["bet_type"]].get(bc.combo_key(b["bet_type"], b["combo"]), 0) * b["yen"] // bc.UNIT for b in bets))
    hm = BM.aligned_mark_types(ents, bets) or {}
    disp_h = next((h for h, m in hm.items() if m == "honmei"), None)
    model_h = int(r.hn[0])
    rec["w4"] = bool(disp_h is not None and disp_h != model_h)
    rec["model_h_in_bets"] = bool(any(model_h in b["combo"] for b in bets))
    marks = {int(h): bc.MARK_NAMES[min(j, len(bc.MARK_NAMES) - 1)] for j, h in enumerate(r.hn)}
    rec["viol"] = ",".join(bc.check_marks([{"bet_type": b["bet_type"], "combo": b["combo"], "yen": b["yen"]} for b in bets], marks, c_min=0.5))
    rec["types"] = ",".join(sorted({b["bet_type"] for b in bets}))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="train")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(bc.OUT_DIR / "w4"))
    a = ap.parse_args()
    inputs = bc.load_inputs()
    per, _ = bc.prepare_periods(inputs, (a.period,), cuts={}, n_boot=10)
    p = per[a.period]
    races = p.races[: a.limit] if a.limit else p.races
    # 連対率（p_top2）は OOF にあれば使い、無ければ勝率と複勝率の中間。合計 2・各馬で 勝率 ≤ 連対率 ≤ 複勝率 にそろえる
    spec = inputs["periods"][a.period]
    import pyarrow.parquet as pq

    t2i = None
    if "p_top2" in pq.read_schema(ROOT / spec["pred"][0]).names:
        t2i = (pd.concat([pd.read_parquet(ROOT / x, columns=["race_id", "horse_id", "p_top2"]) for x in spec["pred"]])
               .assign(race_id=lambda d: d["race_id"].astype(str), horse_id=lambda d: d["horse_id"].astype(str))
               .drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])["p_top2"])
    flat = bc.load_runners(sorted({int(r.race_id[:4]) for r in races}))
    hid = flat.drop_duplicates(["race_id", "horse_number"]).set_index(["race_id", "horse_number"])["horse_id"]
    from src.pipeline.inference.exotic_probabilities import normalize_marginal

    P2 = []
    for r in races:
        v = np.full(len(r.hn), np.nan)
        if t2i is not None:
            v = np.array([t2i.get((r.race_id, str(hid.get((r.race_id, int(h)), ""))), np.nan) for h in r.hn], float)
        if not np.all(np.isfinite(v)):
            v = (r.pw + r.p3) / 2
        v = normalize_marginal(np.minimum(np.maximum(v, r.pw), r.p3), 2.0)
        P2.append(np.asarray(v, float))
    wp = pd.read_parquet(ODDS_WP, columns=["race_id", "horse_number", "place_odds_min", "place_odds_max"])
    wp["race_id"] = wp["race_id"].astype(str)
    ids = {r.race_id for r in races}
    wpd = {k: g for k, g in wp[wp["race_id"].isin(ids)].groupby("race_id")}
    _G.update(races=races, p2=P2, wp=wpd)
    import multiprocessing as mp

    with mp.get_context("fork").Pool(min(a.threads, 4)) as pool:
        recs = pool.map(_one, range(len(races)), chunksize=50)
    d = pd.DataFrame(recs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / f"w4_{a.period}.parquet", index=False)
    b = d[d["cost"] > 0]
    dates = np.array([r.date for r in races])
    idx = {r.race_id: i for i, r in enumerate(races)}
    cost, ret = np.zeros(len(races)), np.zeros(len(races))
    for rid, c, rr in zip(d["race_id"], d["cost"], d["ret"]):
        cost[idx[rid]], ret[idx[rid]] = c, rr
    viol = b["viol"].fillna("")
    summ = {"period": a.period, "races": int(len(d)), "bet_races": int(len(b)), "errors": int((d["status"] == "error").sum()),
            "w4_rate": float(b["w4"].mean()) if len(b) else None,
            "model_honmei_absent_rate": float((~b["model_h_in_bets"].astype(bool)).mean()) if len(b) else None,
            "any_violation_rate": float((viol != "").mean()) if len(b) else None,
            "violation_rates": {k: float(viol.str.contains(k).mean()) for k in ("R2", "R3", "R4", "R5", "R6", "R7", "R8")} if len(b) else {},
            "types": b["types"].value_counts().head(10).to_dict() if len(b) else {},
            "metrics": bc.metrics(cost, ret, dates, bc.poisson_weights(len(races), 1000))}
    (out / f"w4_{a.period}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=bc._np), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=bc._np))


if __name__ == "__main__":
    main()
