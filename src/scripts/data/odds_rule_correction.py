"""ルールベースの補正で、(A) 想定単勝オッズの精度、(B) 勝率→期待値の過大評価、が改善するかを調べる（調査・分析）。

python -m src.scripts.data.odds_rule_correction   # odds_rule_correction.md / .json
A) 想定単勝オッズ（V8 の予測、tp71）: 補正なし／全体の係数／予測オッズ帯ごとの係数／予測順位ごとの係数／レース内で市場らしく正規化（1/オッズの合計＝1/0.8）＋順位ごとの係数。
   年ごとに『前の年だけで係数を当てはめて次の年に当てる』（2023〜2025）。指標は対数オッズの RMSE・MAE と、帯別の偏り。
B) 勝率の補正: 確定単勝オッズの帯ごとに『実際の勝率 ÷ モデルの勝率』の係数を前の年だけで当てはめ、補正した期待値＝補正後の勝率×確定オッズ が、
   実現回収率とどれだけ一致するかを、現行の較正（isotonic、上限つき）と比べる。モデルの勝率は v6e の OOF（2022〜2024）、test 2025、2026 の過去分の算出。
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
ACC = Path(os.environ.get("TP71_ACC", "/home/jovyan/work/tp71_out/acc/rows_single.parquet"))
OOF = Path(os.environ.get("PTRAIN_OOF", ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_z:RB@e.parquet"))
BANDS = [0, 3, 5, 8, 12, 20, 35, 70, 150, 1e9]
BLAB = ["~3", "3~5", "5~8", "8~12", "12~20", "20~35", "35~70", "70~150", "150~"]
TAKE = 0.8


def part_a() -> dict:
    d = pd.read_parquet(ACC)
    d = d[d["bt"] == "tansho"].dropna(subset=["pred", "act"]).copy()
    d = d[(d["pred"] > 0) & (d["act"] > 0)]
    d["lp"], d["la"] = np.log(d["pred"]), np.log(d["act"])
    d["rank"] = d.groupby("race_id")["pred"].rank(method="first").clip(upper=18).astype(int)
    d["band"] = pd.cut(d["pred"], BANDS, labels=BLAB, right=False)
    q = 1.0 / d["pred"]
    d["lp_norm"] = np.log(1.0 / (TAKE * q / q.groupby(d["race_id"]).transform("sum")))      # レース内で 1/オッズの合計が 1/0.8 になるように
    res = {}
    for ty in (2023, 2024, 2025):
        tr, te = d[d["year"] < ty], d[d["year"] == ty].copy()
        off_g = float((tr["la"] - tr["lp"]).median())
        off_b = (tr["la"] - tr["lp"]).groupby(tr["band"], observed=True).median()
        off_r = (tr["la"] - tr["lp"]).groupby(tr["rank"]).median()
        off_nr = (tr["la"] - tr["lp_norm"]).groupby(tr["rank"]).median()
        te["r0"] = te["lp"]
        te["r1"] = te["lp"] + off_g
        te["r2"] = te["lp"] + te["band"].map(off_b).astype(float).fillna(off_g)
        te["r3"] = te["lp"] + te["rank"].map(off_r).fillna(off_g)
        te["r6"] = te["lp_norm"]
        te["r7"] = te["lp_norm"] + te["rank"].map(off_nr).fillna(off_g)
        names = {"r0": "補正なし", "r1": "全体の係数", "r2": "予測オッズ帯ごとの係数", "r3": "予測順位ごとの係数", "r6": "レース内の正規化のみ", "r7": "正規化＋予測順位ごとの係数"}
        row = {}
        for k, nm in names.items():
            e = te[k] - te["la"]
            row[nm] = {"rmse": round(float(np.sqrt((e**2).mean())), 4), "mae": round(float(e.abs().mean()), 4), "bias": round(float(e.mean()), 4),
                       "bias_by_band": {str(b): round(float(g), 3) for b, g in e.groupby(te["band"], observed=True).mean().items()}}
        res[ty] = row
    return res


def load_prob_sets() -> dict[int, pd.DataFrame]:
    sets = {}
    oof = pd.read_parquet(OOF)
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    p25 = pd.read_parquet(ROOT / "data/local/modeling/training_pipeline/final_eval_v6e/test2025_predictions.parquet")
    inf = ROOT / "data/local/inference/predictions_backfill/bundle_v4_2025"
    p26 = pd.concat([pd.read_parquet(f) for f in sorted(inf.glob("*/*.parquet"))], ignore_index=True)
    srcs = {2022: oof, 2023: oof, 2024: oof, 2025: p25, 2026: p26}
    for y, pred in srcs.items():
        pred = pred[["race_id", "horse_id", "p_win"]].copy()
        pred["race_id"], pred["horse_id"] = pred["race_id"].astype(str), pred["horse_id"].astype(str)
        d = ob.load([y])
        raw = pd.read_parquet(ob.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id"])
        raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
        sets[y] = d.merge(raw, on=["race_id", "horse_number"]).merge(pred, on=["race_id", "horse_id"])
    return sets


def blend_p(d: pd.DataFrame, beta: float) -> pd.Series:
    """モデルの勝率と市場の勝率（1/確定オッズをレース内で正規化）を対数で混ぜ、レース内で合計 1 に正規化する。"""
    m = (1.0 / d["odds"]) / (1.0 / d["odds"]).groupby(d["race_id"]).transform("sum")
    lg = beta * np.log(d["p_win"].clip(1e-6)) + (1 - beta) * np.log(m.clip(1e-6))
    e = np.exp(lg - lg.groupby(d["race_id"]).transform("max"))
    return e / e.groupby(d["race_id"]).transform("sum")


def fit_beta(tr: pd.DataFrame) -> float:
    best = (None, 1e9)
    for b in np.arange(0.0, 1.01, 0.1):
        p = blend_p(tr, float(b))
        ll = -np.log(p[tr["finish_position"] == 1].clip(1e-9)).mean()
        if ll < best[1]:
            best = (float(b), ll)
    return best[0]


def part_b() -> dict:
    from src.utils.ev_calibration import calibrated_ev

    cfg = json.loads((ROOT / "config/ev_calibration.json").read_text(encoding="utf-8"))
    S = load_prob_sets()
    res = {}
    for ty in (2023, 2024, 2025, 2026):
        tr = pd.concat([S[y] for y in S if y < ty])
        te = S[ty][S[ty]["ok_win"]].copy()
        tr = tr[tr["ok_win"]]
        tr["won"] = (tr["finish_position"] == 1).astype(float)
        te["won"] = (te["finish_position"] == 1).astype(float)
        tr["band"] = pd.cut(tr["odds"], BANDS, labels=BLAB, right=False)
        te["band"] = pd.cut(te["odds"], BANDS, labels=BLAB, right=False)
        g = tr.groupby("band", observed=True).agg(w=("won", "sum"), p=("p_win", "sum"))
        k = (g["w"] / g["p"]).clip(0.3, 1.5)                                           # 実際の勝ち ÷ モデルの勝率の合計
        te["ev_raw"] = te["p_win"] * te["odds"]
        te["ev_band"] = te["p_win"] * te["band"].map(k).astype(float) * te["odds"]
        te["ev_iso"] = [calibrated_ev(float(e), "win", cfg) for e in te["ev_raw"]]
        beta = fit_beta(tr)
        te["ev_blend"] = blend_p(te, beta) * te["odds"]
        evb_tr = blend_p(tr, beta) * tr["odds"]
        scale = float(tr["ret_win"].mean() / evb_tr.mean())                          # 控除ぶんのずれ（前の年だけで）
        te["ev_blend_s"] = te["ev_blend"] * scale
        row = {"beta": beta, "k_by_band": {str(b): round(float(v), 3) for b, v in k.items()}}
        for nm in ("ev_raw", "ev_iso", "ev_band", "ev_blend", "ev_blend_s"):
            te["bin"] = pd.qcut(te[nm].rank(method="first"), 10, labels=False)
            gg = te.groupby("bin").agg(ev=(nm, "mean"), ret=("ret_win", "mean"))
            row[nm] = {"mean_abs_gap_deciles": round(float((gg["ev"] - gg["ret"]).abs().mean()), 4), "top_decile": [round(float(gg["ev"].iloc[-1]), 3), round(float(gg["ret"].iloc[-1]), 3)],
                       "overall": [round(float(te[nm].mean()), 3), round(float(te["ret_win"].mean()), 3)]}
        res[ty] = row
    return res


def main() -> None:
    a = part_a()
    b = part_b()
    (OUT / "odds_rule_correction.json").write_text(json.dumps({"A": a, "B": b}, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["## A) 想定単勝オッズの補正（対数オッズ。小さいほど良い）", "| 年 | 補正 | RMSE | MAE | 偏り |", "|---|---|---|---|---|"]
    for y, r in a.items():
        for nm, v in r.items():
            L.append(f"| {y} | {nm} | {v['rmse']} | {v['mae']} | {v['bias']} |")
    L += ["", "## B) 期待値の過大評価（十分位ごとの EV と実現回収率の差の平均。小さいほど良い）", "| 年 | 期待値の種類 | 十分位の差の平均 | 最上位十分位 EV/実現 | 全体 EV/実現 |", "|---|---|---|---|---|"]
    for y, r in b.items():
        for nm, lab in (("ev_raw", "生"), ("ev_iso", "現行の較正（isotonic）"), ("ev_band", "オッズ帯ごとの係数"), ("ev_blend", "モデルと市場の対数混合（係数 β 1つ）"), ("ev_blend_s", "混合＋控除の係数（2つの数だけ）")):
            L.append(f"| {y} | {lab} | {r[nm]['mean_abs_gap_deciles']} | {r[nm]['top_decile']} | {r[nm]['overall']} |")
    L += ["", "混合の β（1=モデルのみ、0=市場のみ）: " + ", ".join(f"{y}: {r['beta']:.1f}" for y, r in b.items())]
    L += ["", "係数（実際の勝ち÷モデルの勝率）:"] + [f"- {y}: {r['k_by_band']}" for y, r in b.items()]
    (OUT / "odds_rule_correction.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
