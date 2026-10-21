"""tp-70 較正後の確率が期待値（EV）の計算を良くするか。

python -m src.scripts.data.tp70_ev run --preds tp70_cal_preds.parquet --out DIR

(1) LogLoss の差（較正 − 生／市場）のレース単位ブートストラップ区間 (2) 人気馬の低確率域の予測 vs 実績（生・較正）と識別力（AUC）
(3) EV の信頼性: EV 帯別の『予測 EV と実現回収率』（生・較正・既存の tp-49 単調回帰） (4) 同じ購入数で EV の高い順に買った実現回収率。
単勝は 2023〜2025・2026、複勝は払戻のある 2023〜2025。評価は前年までで当てはめた較正器（tp70_fit compare の出力）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.scripts.data.tp70_lib import HIT, MKT, PRED, TARGETS, logloss, pop_class, race_boot_ci
from src.utils.ev_calibration import apply_spec, load_config

CAL = "C3"
ALT = ["C3", "C5", "C6", "C3r"]
EV_BINS = [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 99]


def col(k: str, t: str) -> str:
    return PRED[t] if k == "C0" else (MKT[t] if k == "C1" else f"cal_{k}_{t}")


def ll_diff(df: pd.DataFrame, t: str, a: str, b: str) -> float:
    return logloss(df[HIT[t]], df[col(a, t)]) - logloss(df[HIT[t]], df[col(b, t)])


def run(args) -> None:
    d = pd.read_parquet(args.preds)
    d = d[d["year"] >= 2023].copy()
    d["pc"] = pop_class(d["pop"]).astype(str)
    res: dict = {}
    # (1) LogLoss 差の区間
    rows = []
    for per, ys in (("2023-25", [2023, 2024, 2025]), ("2026", [2026])):
        x = d[d["year"].isin(ys)].reset_index(drop=True)
        for t in TARGETS:
            for k in ALT:
                for ref in ("C0", "C1"):
                    m, lo, hi = race_boot_ci(x, lambda z, t=t, k=k, ref=ref: ll_diff(z, t, k, ref), reps=200)
                    rows.append({"period": per, "target": t, "method": k, "vs": ref, "diff": m, "lo": lo, "hi": hi})
    res["ll_diff"] = pd.DataFrame(rows).round(5).to_dict("records")
    # (2) 人気馬（1〜5）の低確率域: 生 / 較正 / 実績 と AUC
    low = []
    pop = d[d["pop"] <= 5]
    for per, ys in (("2023-25", [2023, 2024, 2025]), ("2026", [2026])):
        x = pop[pop["year"].isin(ys)]
        for lo_, hi_ in ((0, 0.10), (0.10, 0.15), (0.15, 0.22), (0.22, 0.30), (0.30, 1.0)):
            g = x[(x["p_top3"] > lo_) & (x["p_top3"] <= hi_)]
            if len(g) < 30:
                continue
            low.append({"period": per, "raw_bin": f"({lo_},{hi_}]", "n": len(g), "raw": g["p_top3"].mean(), "C3": g["cal_C3_top3"].mean(), "C6": g["cal_C6_top3"].mean(), "mkt": g["m_top3"].mean(), "act": g["h3"].mean()})
        for t in TARGETS:
            low.append({"period": per, "auc_" + t: {k: float(roc_auc_score(x[HIT[t]], x[col(k, t)])) for k in ("C0", "C1", "C3", "C6")}, "n": len(x)})
    res["popular_low"] = pd.DataFrame(low).to_dict("records")
    # 全馬の AUC（識別力が落ちていないか）
    auc = {}
    for per, ys in (("2023-25", [2023, 2024, 2025]), ("2026", [2026])):
        x = d[d["year"].isin(ys)]
        auc[per] = {t: {k: float(roc_auc_score(x[HIT[t]], x[col(k, t)])) for k in ("C0", "C1", "C3", "C6")} for t in TARGETS}
    res["auc_all"] = auc
    # (3)(4) EV の信頼性と購入
    cfg = load_config()
    iso = (cfg.get("kinds") or {}).get("win")
    plc = (cfg.get("kinds") or {}).get("place")
    d["odds"] = d["win_odds"]
    d["po"] = d["place_odds_min"] + 0.3 * (d["place_odds_max"] - d["place_odds_min"])
    ev_rows, sel_rows = [], []
    for kind in ("win", "place"):
        t = "win" if kind == "win" else "top3"
        ret = (d["ret_win"] if kind == "win" else d["ret_place"]) / 100.0
        o = d["odds"] if kind == "win" else d["po"]
        ok = d["year"].between(2023, 2026) if kind == "win" else d["year"].between(2023, 2025)
        for per, ys in (("2023-25", [2023, 2024, 2025]), ("2026", [2026])):
            msk = ok & d["year"].isin(ys) & o.notna() & ret.notna()
            if msk.sum() == 0:
                continue
            x = d[msk].copy()
            x["ret"] = ret[msk]
            x["o"] = o[msk]
            evs = {"raw": x[PRED[t]] * x["o"], "C3": x[f"cal_C3_{t}"] * x["o"], "C6": x[f"cal_C6_{t}"] * x["o"], "mkt": x[MKT[t]] * x["o"]}
            spec = iso if kind == "win" else plc
            if spec:
                evs["tp49(生EVの単調回帰)"] = pd.Series(apply_spec(spec, evs["raw"].to_numpy()), index=x.index)
            for name, ev in evs.items():
                b = pd.cut(ev, EV_BINS)
                g = x.assign(ev=ev, b=b).groupby("b", observed=True).agg(n=("ret", "size"), ev=("ev", "mean"), real=("ret", "mean"))
                for bi, r in g.iterrows():
                    ev_rows.append({"kind": kind, "period": per, "ev_type": name, "bin": str(bi), "n": int(r["n"]), "ev": round(r["ev"], 4), "real": round(r["real"], 4)})
                # 信頼性の要約: 件数加重の |予測 EV − 実現| と、実現 = a + b×予測 の傾き
                w = g["n"] / g["n"].sum()
                mae = float((w * (g["ev"] - g["real"]).abs()).sum())
                slope = float(np.polyfit(g["ev"], g["real"], 1, w=np.sqrt(g["n"]))[0]) if len(g) >= 3 else float("nan")
                sel_rows.append({"kind": kind, "period": per, "ev_type": name, "metric": "ev_gap_mae", "value": round(mae, 4), "slope": round(slope, 3)})
            # 同じ購入数での実現回収率: EV の高い順に上位 q% を購入（全期間プール、1 レース内の順位は問わない）
            for q in (0.01, 0.02, 0.05):
                nbet = int(len(x) * q)
                for name, ev in evs.items():
                    top = x.assign(ev=ev).nlargest(nbet, "ev")
                    mean, lo, hi = race_boot_ci(top.reset_index(drop=True), lambda z: float(z["ret"].mean()), reps=200)
                    sel_rows.append({"kind": kind, "period": per, "ev_type": name, "metric": f"roi_top{int(q * 100)}pct", "value": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": nbet, "hit": round(float((top["ret"] > 0).mean()), 4), "ev_mean": round(float(top["ev"].mean()), 4)})
            # 閾値買い（EV ≥ 1.0 / 1.1 / 1.2）
            for th in (1.0, 1.1, 1.2):
                for name, ev in evs.items():
                    top = x.assign(ev=ev)
                    top = top[top["ev"] >= th]
                    if len(top) < 30:
                        sel_rows.append({"kind": kind, "period": per, "ev_type": name, "metric": f"roi_ev{th}", "value": None, "n": len(top)})
                        continue
                    mean, lo, hi = race_boot_ci(top.reset_index(drop=True), lambda z: float(z["ret"].mean()), reps=200)
                    sel_rows.append({"kind": kind, "period": per, "ev_type": name, "metric": f"roi_ev{th}", "value": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(top), "hit": round(float((top["ret"] > 0).mean()), 4), "ev_mean": round(float(top["ev"].mean()), 4)})
    res["ev_bins"] = ev_rows
    res["selection"] = sel_rows
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tp70_ev.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    s = pd.DataFrame(sel_rows)
    print(s[s["metric"] == "ev_gap_mae"].to_string(index=False))
    print(s[s["metric"].str.startswith("roi_top")].to_string(index=False))
    print(pd.DataFrame(res["ll_diff"]).query("method=='C3'").to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--preds", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
