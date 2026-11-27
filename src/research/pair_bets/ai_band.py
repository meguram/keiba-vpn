"""AI ÷ LBS の帯ごとの、組の市場 m ÷ 単勝からの組の確率（LBS）。年ごと（着順・払戻は読まない）。

    KEIBA_SESSION=pair-bets nice -n 19 python -m src.research.pair_bets.ai_band

- 予測と判定の線は数える前に固定した（docs/operations/pair_bets/predictions_ai_band.json・commit 42a296a3）。
- 帯: x = log(AI の組の確率 ÷ 単勝からの組の確率)（どちらも LBS）。組のオッズ m は帯に使わない。年 × 券種の中の十分位。
- 組・m の作りは market.py と同じ（平場・ワイドの m は中点の逆数を和 3）。81 番人気より下の馬連を除いた版も出す。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pair_bets.market import LBS, OUT, horses, pair_probs, pairs

DST = OUT.parent / "ai_band.json"
PRED = OUT.parent / "predictions_ai_band.json"


def collect(ty: str, H: pd.DataFrame) -> pd.DataFrame:
    hg = {rid: g for rid, g in H.groupby("race_id", sort=False)}
    F = pairs(ty)
    F = F[F["race_id"].isin(hg.keys())]
    rows = []
    for rid, g in F.groupby("race_id", sort=False):
        h = hg[rid]
        if len(h) < 5:
            continue
        pos = {int(x): i for i, x in enumerate(h["hn"].to_numpy())}
        ia, ib = g["a"].map(pos), g["b"].map(pos)
        ok = ia.notna() & ib.notna()
        if ok.sum() < len(g) * 0.9:
            continue
        g = g[ok]
        ia, ib = ia[ok].astype(int).to_numpy(), ib[ok].astype(int).to_numpy()
        inv = 1 / g["o_mid"].to_numpy(float)
        m = inv / inv.sum() * (3.0 if ty == "wide" else 1.0)
        uq, wq = pair_probs(h["q"].to_numpy(float), LBS)
        up, wp = pair_probs(h["p"].to_numpy(float), LBS)
        Q, P = (uq, up) if ty == "umaren" else (wq, wp)
        rows.append(pd.DataFrame({"race_id": rid, "year": int(h["year"].iloc[0]), "pop": g["popularity"].to_numpy(), "m": m,
                                  "hq": Q[ia, ib], "hp": P[ia, ib]}))
    return pd.concat(rows, ignore_index=True)


def table(R: pd.DataFrame) -> dict:
    out = {}
    for y, g in R.groupby("year"):
        x = np.log(g["hp"].clip(1e-12) / g["hq"].clip(1e-12))
        dec = pd.qcut(x.rank(method="first"), 10, labels=False)
        s = g.assign(dec=dec).groupby("dec").agg(m=("m", "sum"), hq=("hq", "sum"), n=("m", "size"), hp=("hp", "sum"))
        ratio = (s["m"] / s["hq"]).round(4).tolist()
        ai = (s["hp"] / s["hq"]).round(4).tolist()
        out[str(int(y))] = {"m_over_lbs": ratio, "ai_over_lbs": ai, "n": s["n"].astype(int).tolist(), "top_over_bottom": round(ratio[-1] / ratio[0], 4)}
    return out


def verdict(t: dict) -> str:
    v = [t[y]["top_over_bottom"] for y in sorted(t)]
    if all(0.97 <= x <= 1.03 for x in v):
        return "P1"
    if all(x > 1.03 for x in v):
        return "P2"
    if all(x < 0.97 for x in v):
        return "P3"
    return "決まらない"


def main() -> None:
    from src.research.pair_bets.pb_ledger import open_token
    with open_token(tool="src/research/pair_bets/ai_band.py", version="pair_bets_ai_band", period=("2022-01-01", "2024-12-31"),
                    purpose="select", reads=("確定オッズ", "AI の予測"),
                    reason="1 号の依頼: m を使わない AI ÷ LBS の帯の m ÷ LBS（予測は 42a296a3 で先に固定）。着順・払戻は読まない") as tok:
        H = horses()
        res = {"predictions": json.loads(PRED.read_text(encoding="utf-8"))["name"], "bands": "AI ÷ LBS の十分位（1 = 低い）"}
        for ty in ("umaren", "wide"):
            R = collect(ty, H)
            t = table(R)
            res[ty] = {"all": t, "verdict": verdict(t)}
            if ty == "umaren":
                t2 = table(R[pd.to_numeric(R["pop"], errors="coerce") <= 80])
                res[ty]["pop_le_80"] = t2
                res[ty]["verdict_pop_le_80"] = verdict(t2)
        DST.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        tok.finish(result_path=DST)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
