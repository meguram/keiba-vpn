"""★（較正後の単勝期待値 ≥ 0.95、想定単勝 10〜30 倍、1 レース 1 頭）が出るレースは賭ける価値があるか。条件別（頭数・クラス・面・馬場・期待値の水準・オッズ帯）の回収率（調査・分析）。

python -m src.scripts.data.star_worth   # star_worth.md / .json
2022〜2026 年を合算（モデルの勝率: 2022〜24 は v6e の OOF、2025 は test 2025、2026 は過去分の算出）。確定オッズ・確定払戻。区間はレース単位ブートストラップ。
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
TH = float(os.environ.get("STAR_TH", 0.95))
MIN_FIELD = int(os.environ.get("STAR_MIN_FIELD", 0))


def main() -> None:
    from src.utils.ev_calibration import calibrated_ev

    cfg = json.loads((ROOT / "config/ev_calibration.json").read_text(encoding="utf-8"))
    S = oc.load_prob_sets()
    d = pd.concat([S[y] for y in sorted(S)], ignore_index=True)
    meta = pd.concat([pd.read_parquet(ob.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "grade", "race_class", "track_condition", "venue"]).drop_duplicates("race_id") for y in S])
    meta["race_id"] = meta["race_id"].astype(str)
    d = d.merge(meta, on="race_id", how="left")
    d["ev_cal"] = [calibrated_ev(float(p * o), "win", cfg) for p, o in zip(d["p_win"], d["odds"])]
    b = d[(d["odds"] >= 10) & (d["odds"] < 30) & (d["field_size"] >= MIN_FIELD)]
    s = b[b["ev_cal"] >= TH].sort_values("ev_cal", ascending=False).drop_duplicates("race_id")
    sw, sp = s[s["ok_win"]].copy(), s[s["ok_place"]].copy()
    N = d["race_id"].nunique()
    rows = []

    def add(label: str, g: pd.DataFrame, gp: pd.DataFrame) -> None:
        if g["race_id"].nunique() < 150:
            return
        w = boot(g, "ret_win", 300); p = boot(gp, "ret_place", 300) if len(gp) > 100 else (np.nan,) * 3
        yr = g.groupby("year")["ret_win"].mean()
        rows.append({"条件": label, "★のレース": int(g["race_id"].nunique()), "的中率": round(float((g["ret_win"] > 0).mean()), 3), "単勝回収": round(w[0], 3), "区間": [round(w[1], 2), round(w[2], 2)], "複勝回収": round(p[0], 3),
                     "年別単勝": {int(a): round(float(c), 2) for a, c in yr.items()}})

    add("全体", sw, sp)
    sw["fs"] = pd.cut(sw["field_size"], [0, 12, 15, 99], labels=["~12頭", "13~15頭", "16頭~"]); sp["fs"] = pd.cut(sp["field_size"], [0, 12, 15, 99], labels=["~12頭", "13~15頭", "16頭~"])
    sw["evb"] = pd.cut(sw["ev_cal"], [0, 1.0, 1.1, 9], labels=["0.95~1.0", "1.0~1.1", "1.1~"]); sp["evb"] = pd.cut(sp["ev_cal"], [0, 1.0, 1.1, 9], labels=["0.95~1.0", "1.0~1.1", "1.1~"])
    sw["ob"] = pd.cut(sw["odds"], [10, 15, 20, 30], labels=["10~15倍", "15~20倍", "20~30倍"], right=False); sp["ob"] = pd.cut(sp["odds"], [10, 15, 20, 30], labels=["10~15倍", "15~20倍", "20~30倍"], right=False)
    for col, nm in (("surface", "面"), ("fs", "頭数"), ("evb", "較正後の期待値"), ("ob", "想定オッズ"), ("grade", "格"), ("track_condition", "馬場")):
        for v in sorted(sw[col].dropna().unique(), key=str):
            add(f"{nm}={v}", sw[sw[col] == v], sp[sp[col] == v])
    pd.DataFrame(rows).to_json(OUT / "star_worth.json", orient="records", force_ascii=False, indent=1)
    L = [f"全 {N} レースのうち★が出るレース {sw['race_id'].nunique()}（{sw['race_id'].nunique() / N:.1%}）", "| 条件 | ★のレース | 的中率 | 単勝回収(95%区間) | 複勝回収 | 年別単勝 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['条件']} | {r['★のレース']} | {r['的中率']} | {r['単勝回収']} ({r['区間'][0]}~{r['区間'][1]}) | {r['複勝回収']} | {r['年別単勝']} |")
    (OUT / f"star_worth{MIN_FIELD or ''}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
