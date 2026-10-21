"""tp-70 条件別（頭数帯・競馬場・月）の『較正前後のずれ』。

python -m src.scripts.data.tp70_cond run --preds tp70_cal_preds.parquet --out DIR
各条件で、複勝確率の 平均の差（実績−予測）と LogLoss の差（手法 − 市場）を 生 / C3 / C6 / C3r で並べる。評価は前年までで当てはめた較正器の出力。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.scripts.data.tp70_lib import HIT, MKT, PRED, logloss

M = {"raw": "p_top3", "mkt": "m_top3", "C3": "cal_C3_top3", "C6": "cal_C6_top3", "C3r": "cal_C3r_top3"}


def run(a) -> None:
    d = pd.read_parquet(a.preds)
    d = d[d["year"] >= 2023].copy()
    d["period"] = d["year"].map(lambda y: "2026" if y == 2026 else "2023-25")
    d["field"] = pd.cut(d["n"], [7, 11, 14, 16, 18, 30], labels=["8-11頭", "12-14頭", "15-16頭", "17-18頭", "19頭以上"]).astype(str)
    d["season"] = pd.cut(d["month"], [0, 2, 5, 8, 11, 12], labels=["冬(1-2)", "春(3-5)", "夏(6-8)", "秋(9-11)", "12月"], ordered=False).astype(str)
    rows = []
    for key in ("field", "venue", "season", "month"):
        for (per, v), g in d.groupby(["period", key]):
            if len(g) < 1500:
                continue
            ll_m = logloss(g["h3"], g[M["mkt"]])
            r = {"cond": key, "value": str(v), "period": per, "n": len(g)}
            for k, c in M.items():
                r[f"bias_{k}"] = float(g["h3"].mean() - g[c].mean())
                r[f"llgap_{k}"] = logloss(g["h3"], g[c]) - ll_m
            rows.append(r)
    out = Path(a.out)
    (out / "tp70_cond.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    x = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(x[x["cond"].isin(["field", "season"])][["cond", "value", "period", "n", "bias_raw", "bias_C3", "llgap_raw", "llgap_C3", "llgap_C6", "llgap_C3r"]].round(4).to_string(index=False))
    print(x[x["cond"] == "month"][["value", "period", "n", "llgap_raw", "llgap_C3", "llgap_C3r"]].round(4).to_string(index=False))


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
