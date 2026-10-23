"""めぐ印★（おすすめ穴馬）の対象に最適な単勝オッズ帯を探す（調査・分析）。モデル（v6e の OOF、2022〜2024 年）で『期待値が高い』と見た馬だけを帯ごとに買ったときの回収率。

python -m src.scripts.data.star_band_ev   # star_band_ev.md / .json
選ぶ規則: 帯ごとに ev_win = 勝率 × 確定単勝オッズ が閾値以上の馬（各レースで帯内の最大 1 頭と、閾値以上の全頭の 2 通り）。
回収率は確定払戻（単勝・複勝、100 円）。区間はレース単位ブートストラップ。年別（2022・23・24）の符号も見る。確定オッズで代用。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import odds_band_ev as ob

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
OOF = Path(os.environ.get("PTRAIN_OOF", ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_z:RB@e.parquet"))
BANDS = [(5, 7), (7, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100), (10, 30), (7, 20), (10, 50)]
THRESH = [1.0, 1.2, 1.5]


def boot(g: pd.DataFrame, ret: str, n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(g["race_id"].to_numpy())
    s = np.bincount(codes, weights=g[ret].to_numpy()); n = np.bincount(codes)
    b = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(uniq), len(uniq)) for _ in range(n_boot))]
    return float(g[ret].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    d = ob.load([2022, 2023, 2024])
    oof = pd.read_parquet(OOF)
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    raw = pd.concat([pd.read_parquet(ob.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id"]) for y in (2022, 2023, 2024)])
    raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
    d = d.merge(raw, on=["race_id", "horse_number"], how="left").merge(oof[["race_id", "horse_id", "p_win", "p_top3"]], on=["race_id", "horse_id"], how="inner")
    d["ev_win"] = d["p_win"] * d["odds"]
    rows = []
    for lo, hi in BANDS:
        b = d[(d["odds"] >= lo) & (d["odds"] < hi)]
        for th in THRESH:
            sel_all = b[b["ev_win"] >= th]
            sel_one = sel_all.sort_values("ev_win", ascending=False).drop_duplicates("race_id")
            for mode, g in (("閾値以上の全頭", sel_all), ("レースごとに最大1頭", sel_one)):
                gw, gp = g[g["ok_win"]], g[g["ok_place"]]
                if len(gw) < 300:
                    continue
                w = boot(gw, "ret_win"); p = boot(gp, "ret_place") if len(gp) > 300 else (np.nan,) * 3
                yr = gw.groupby("year")["ret_win"].mean(); yp = gp.groupby("year")["ret_place"].mean()
                base_w = float(b[b["ok_win"]]["ret_win"].mean()); base_p = float(b[b["ok_place"]]["ret_place"].mean())
                rows.append({"帯": f"{lo}~{hi}", "規則": mode, "閾値": th, "出走": int(len(gw)), "レース": int(gw["race_id"].nunique()), "的中率": float((gw["ret_win"] > 0).mean()), "複勝的中率": float((gp["ret_place"] > 0).mean()),
                             "単勝": w[0], "単勝区間": [w[1], w[2]], "複勝": p[0], "複勝区間": [p[1], p[2]], "帯全体の単勝": base_w, "帯全体の複勝": base_p,
                             "年別単勝": {int(a): round(float(c), 3) for a, c in yr.items()}, "年別複勝": {int(a): round(float(c), 3) for a, c in yp.items()}})
    pd.DataFrame(rows).to_json(OUT / "star_band_ev.json", orient="records", force_ascii=False, indent=1)
    L = ["| 帯 | 規則 | EV閾値 | 出走 | 単勝的中 | 単勝回収(95%区間) | 複勝的中 | 複勝回収(95%区間) | 帯全体 単/複 | 年別単勝 | 年別複勝 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['帯']} | {r['規則']} | {r['閾値']} | {r['出走']} | {r['的中率']:.3f} | {r['単勝']:.3f} ({r['単勝区間'][0]:.2f}~{r['単勝区間'][1]:.2f}) | {r['複勝的中率']:.3f} | {r['複勝']:.3f} ({r['複勝区間'][0]:.2f}~{r['複勝区間'][1]:.2f}) | {r['帯全体の単勝']:.3f}/{r['帯全体の複勝']:.3f} | {list(r['年別単勝'].values())} | {list(r['年別複勝'].values())} |")
    (OUT / "star_band_ev.md").write_text("\n".join(L), encoding="utf-8")
    print(len(rows), "rows")


def calibrated_main() -> None:
    """本番と同じ較正後の期待値（config/ev_calibration.json の win、上限あり）で★の規則を評価する。過大評価の生の期待値は使わない。"""
    from src.utils.ev_calibration import calibrated_ev, load_config

    cfg = json.loads(Path(os.environ.get("PTRAIN_EVCFG", ROOT / "config/ev_calibration.json")).read_text(encoding="utf-8"))
    d = ob.load([2022, 2023, 2024])
    oof = pd.read_parquet(OOF)
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    raw = pd.concat([pd.read_parquet(ob.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id"]) for y in (2022, 2023, 2024)])
    raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
    d = d.merge(raw, on=["race_id", "horse_number"]).merge(oof[["race_id", "horse_id", "p_win"]], on=["race_id", "horse_id"])
    d["ev_raw"] = d["p_win"] * d["odds"]
    d["ev_cal"] = [calibrated_ev(float(e), "win", cfg) for e in d["ev_raw"]]
    N, days = d["race_id"].nunique(), d["date"].nunique()
    rows = []
    for lo, hi in ((7, 20), (10, 30), (7, 30), (7, 10), (10, 15), (15, 20), (20, 30)):
        b = d[(d["odds"] >= lo) & (d["odds"] < hi)]
        for th in (0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
            s = b[b["ev_cal"] >= th].sort_values(["ev_cal", "ev_raw"], ascending=False).drop_duplicates("race_id")
            sw, sp = s[s["ok_win"]], s[s["ok_place"]]
            if len(sw) < 300:
                continue
            w, p = boot(sw, "ret_win"), boot(sp, "ret_place")
            yr = sw.groupby("year")["ret_win"].mean()
            rows.append({"帯": f"{lo}~{hi}", "較正後EV閾値": th, "★が出るレース": int(s["race_id"].nunique()), "割合": s["race_id"].nunique() / N, "何レースに1回": N / s["race_id"].nunique(), "1日あたり": s["race_id"].nunique() / days,
                         "単勝的中": float((sw["ret_win"] > 0).mean()), "単勝回収": w[0], "区間": [w[1], w[2]], "複勝回収": p[0], "年別単勝": [round(float(x), 3) for x in yr.values], "実現/較正後EV": float(sw["ret_win"].mean() / sw["ev_cal"].mean())})
    L = ["| 帯 | 較正後EV閾値 | ★が出るレース | 割合 | 何レースに1回 | 1日あたり | 単勝的中 | 単勝回収(95%区間) | 複勝回収 | 年別単勝 | 実現÷較正後EV |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['帯']} | {r['較正後EV閾値']} | {r['★が出るレース']} | {r['割合']:.3f} | {r['何レースに1回']:.1f} | {r['1日あたり']:.1f} | {r['単勝的中']:.3f} | {r['単勝回収']:.3f} ({r['区間'][0]:.2f}~{r['区間'][1]:.2f}) | {r['複勝回収']:.3f} | {r['年別単勝']} | {r['実現/較正後EV']:.2f} |")
    (OUT / "star_band_ev_calibrated.md").write_text("\n".join(L), encoding="utf-8")
    print(f"races {N}, days {days}, rows {len(rows)}")



def holdout_main() -> None:
    """2025 年（v6e の test 2025 予測）と 2026 年（v4 の過去分の算出）で、固定した規則（較正後の期待値の閾値）の頻度と回収率を確認する。閾値は 2022〜2024 年で決めた。"""
    from src.utils.ev_calibration import calibrated_ev

    cfg = json.loads(Path(os.environ.get("PTRAIN_EVCFG", ROOT / "config/ev_calibration.json")).read_text(encoding="utf-8"))
    inf = ROOT / "data/local/inference/predictions_backfill/bundle_v4_2025"
    p25 = pd.read_parquet(ROOT / "data/local/modeling/training_pipeline/final_eval_v6e/test2025_predictions.parquet")
    p26 = pd.concat([pd.read_parquet(f) for f in sorted(inf.glob("*/*.parquet"))], ignore_index=True)
    res = {}
    for yr, pred in ((2025, p25), (2026, p26)):
        pred = pred[["race_id", "horse_id", "p_win"]].copy()
        pred["race_id"], pred["horse_id"] = pred["race_id"].astype(str), pred["horse_id"].astype(str)
        d = ob.load([yr])
        raw = pd.read_parquet(ob.TABLES / str(yr) / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id"])
        raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
        d = d.merge(raw, on=["race_id", "horse_number"]).merge(pred, on=["race_id", "horse_id"])
        d["ev_cal"] = [calibrated_ev(float(p * o), "win", cfg) for p, o in zip(d["p_win"], d["odds"])]
        N, days = d["race_id"].nunique(), d["date"].nunique()
        for lo, hi in ((7, 20), (10, 30), (7, 30)):
            b = d[(d["odds"] >= lo) & (d["odds"] < hi)]
            for th in (0.9, 0.95, 1.0):
                s = b[b["ev_cal"] >= th].sort_values("ev_cal", ascending=False).drop_duplicates("race_id")
                sw, sp = s[s["ok_win"]], s[s["ok_place"]]
                if len(sw) < 50:
                    continue
                w, p = boot(sw, "ret_win"), boot(sp, "ret_place")
                res[f"{yr} {lo}~{hi} EV{th}"] = {"races": N, "stars": int(s["race_id"].nunique()), "share": round(s["race_id"].nunique() / N, 3), "per_star_races": round(N / s["race_id"].nunique(), 1), "per_day": round(s["race_id"].nunique() / days, 1),
                                                  "win_ret": round(w[0], 3), "win_ci": [round(w[1], 2), round(w[2], 2)], "place_ret": round(p[0], 3), "band_win_ret": round(float(b[b["ok_win"]]["ret_win"].mean()), 3)}
    (OUT / "star_band_ev_holdout.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    import sys

    if "--calibrated" in sys.argv:
        calibrated_main()
    elif "--holdout" in sys.argv:
        holdout_main()
    else:
        main()
