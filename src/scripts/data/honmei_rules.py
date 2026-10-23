"""◎（本命）の決め方の比較（調査・分析）。モデルの勝率 1 位、v2 の規則（勝率上位 3 頭のうち 均等確率×2 以上で、生の期待値が最大）、
同じ規則を較正後の期待値にしたもの、など。勝率・単勝/複勝の回収率・1 番人気と同じ馬になる割合を、期間別に比べる。

python -m src.scripts.data.honmei_rules   # honmei_rules.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import odds_band_ev as ob
from src.scripts.data import odds_rule_correction as oc
from src.scripts.data.star_band_ev import boot

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))


def pick(d: pd.DataFrame, topk: int, mult: float, evcol: str | None) -> pd.DataFrame:
    n = d.groupby("race_id")["p_win"].transform("size")
    d = d.assign(base=1.0 / n, rk=d.groupby("race_id")["p_win"].rank(ascending=False, method="first"))
    pool = d[(d["rk"] <= topk) & (d["p_win"] >= d["base"] * mult)]
    if evcol is None:
        return d[d["rk"] == 1]
    got = pool.sort_values([evcol, "p_win"], ascending=False).drop_duplicates("race_id")
    miss = d[(d["rk"] == 1) & ~d["race_id"].isin(got["race_id"])]            # 該当なしは勝率 1 位
    return pd.concat([got, miss])


def main() -> None:
    from src.utils.ev_calibration import calibrated_ev

    cfg = json.loads((ROOT / "config/ev_calibration.json").read_text(encoding="utf-8"))
    S = oc.load_prob_sets()
    rules = {"勝率1位": (3, 0, None), "v2: 上位3頭・均等×2以上・生の期待値最大": (3, 2.0, "ev_raw"), "上位3頭・均等×2以上・較正後の期待値最大": (3, 2.0, "ev_cal"),
             "上位3頭・均等×1以上・較正後の期待値最大": (3, 1.0, "ev_cal"), "上位2頭・均等×2以上・較正後の期待値最大": (2, 2.0, "ev_cal"), "上位5頭・均等×1.5以上・較正後の期待値最大": (5, 1.5, "ev_cal")}
    rows = []
    for nm, ys in (("2022-24 OOF", [2022, 2023, 2024]), ("2025", [2025]), ("2026", [2026])):
        d = pd.concat([S[y] for y in ys])
        d = d[d["ok_win"]].copy()
        d["won"] = (d["finish_position"] == 1).astype(float)
        d["ev_raw"] = d["p_win"] * d["odds"]
        d["ev_cal"] = [calibrated_ev(float(e), "win", cfg) for e in d["ev_raw"]]
        fav = d.assign(fr=d.groupby("race_id")["odds"].rank(method="first"))
        favid = fav[fav["fr"] == 1].set_index("race_id")["horse_id"]
        for rn, (k, m, ev) in rules.items():
            g = pick(d, k, m, ev)
            w = boot(g, "ret_win", 200)
            gp = g[g["ok_place"]]
            rows.append({"期間": nm, "規則": rn, "レース": int(g["race_id"].nunique()), "勝率": round(float(g["won"].mean()), 4), "単勝回収": round(w[0], 3), "区間": [round(w[1], 2), round(w[2], 2)],
                         "複勝回収": round(float(gp["ret_place"].mean()), 3), "1番人気と同じ": round(float((g.set_index("race_id")["horse_id"] == favid.reindex(g["race_id"]).values).mean()), 3), "平均オッズ": round(float(g["odds"].mean()), 1)})
    L = ["| 期間 | 規則 | レース | 勝率 | 単勝回収(95%区間) | 複勝回収 | 1番人気と同じ | 平均オッズ |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['期間']} | {r['規則']} | {r['レース']} | {r['勝率']} | {r['単勝回収']} ({r['区間'][0]}~{r['区間'][1]}) | {r['複勝回収']} | {r['1番人気と同じ']} | {r['平均オッズ']} |")
    (OUT / "honmei_rules.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
