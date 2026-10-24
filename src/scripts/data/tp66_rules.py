"""tp-66 マーキング規則の比較（tp66_eval rules から呼ぶ。ホストで実行）。

選んだ指標（--score、既定は環境変数 TP66_SCORE または max5_margin）について、印の付け方（レース内の上位 k 頭 / 突破ライン超え / z の閾値 /
V5 が低評価の馬だけ）を比べる。期間ごと（選択 2022〜23・確認 2024・最終 2025・テストセット② 2026）に、レースあたりの印の数、
印の馬の 3 着内率・勝率・倍率（ランダムに比べた比）・複勝/単勝回収率・人気薄の好走の割合を出し、印あり − 印なしの倍率差の 95% 区間（レース単位）を付ける。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from src.scripts.data import tp66_eval as E
from src.scripts.data import tp66_lib as L


def lift_ci(g: pd.DataFrame, flag: pd.Series, n_boot: int = 1000, seed: int = 0) -> list[float] | None:
    """(印ありの 3 着内倍率) − (印なしの 3 着内倍率) の 95% 区間。倍率 = 実績の合計 ÷ ランダムな期待の合計。"""
    f = flag.astype(bool).to_numpy()
    agg = pd.DataFrame({"race_id": g["race_id"].to_numpy(), "am": np.where(f, g["top3"], 0.0), "ae": np.where(f, g["exp_t3"], 0.0),
                        "bm": np.where(~f, g["top3"], 0.0), "be": np.where(~f, g["exp_t3"], 0.0)}).groupby("race_id").sum()
    if len(agg) < 30:
        return None
    a = agg.to_numpy()
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(len(a), np.full(len(a), 1.0 / len(a)), size=n_boot)
    s = cnt @ a
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    d = s[ok, 0] / s[ok, 1] - s[ok, 2] / s[ok, 3]
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def build_rules(df: pd.DataFrame, score: str, margin: str | None) -> dict[str, pd.Series]:
    rules: dict[str, pd.Series] = {}
    df = df.copy()
    df["z"] = L.within_race_z(df, score)
    above = df[margin].ge(0) if margin and margin in df else pd.Series(False, index=df.index)
    v5_rank = df.groupby("race_id")["m_top3"].rank(ascending=False, method="first")
    df["_above"] = above
    for k in (1, 2, 3):
        rules[f"top{k}"] = L.mark_top_k(df, score, k)
        if margin:
            rules[f"top{k}&above"] = L.mark_top_k(df, score, k, require="_above")
    if margin:
        rules["above(all)"] = above & df[score].notna()
    for z in (1.0, 1.5, 2.0):
        rules[f"z>={z}"] = df["z"].ge(z)
    for k in (2, 3):
        rules[f"top{k}&V5rank>3"] = L.mark_top_k(df, score, k) & v5_rank.gt(3)
        rules[f"top{k}&V5rank>5"] = L.mark_top_k(df, score, k) & v5_rank.gt(5)
    rules["top1&V5top3"] = L.mark_top_k(df, score, 1) & v5_rank.le(3)
    if margin:
        a = rules["top1&above"]
        rules["tierA(top1&above)"] = a
        rules["tierB(rank2-3&above)"] = rules["top3&above"] & ~a
        rules["tierA&z>=1.0"] = a & df["z"].ge(1.0)
        rules["tierA&z<1.0"] = a & df["z"].lt(1.0)
        rules["tierA|B"] = rules["top3&above"]
    return rules


def main() -> int:
    score = os.environ.get("TP66_SCORE", "max5_margin")
    margin = os.environ.get("TP66_MARGIN", score if score.endswith("_margin") else "")
    feat = pd.read_parquet(E.OUT / "feat.parquet")
    df = E.attach_outcomes(feat)
    df = df[df["n_in_race"] >= 6].reset_index(drop=True)
    out: dict = {"score": score, "margin": margin or None, "rules": {}}
    rules = build_rules(df, score, margin or None)
    for name, flag in rules.items():
        row = {}
        for p in E.PERIODS:
            m = E.period_mask(df, p)
            g = df[m]
            f = flag[m]
            if not len(g):
                continue
            st = E.mark_stats(g, f)
            st["marks_per_race"] = round(float(f.sum() / max(g["race_id"].nunique(), 1)), 3)
            st["race_cover"] = round(float(g[f]["race_id"].nunique() / max(g["race_id"].nunique(), 1)), 3)
            st["diff_lift_ci"] = lift_ci(g, f)
            row[p] = st
        out["rules"][name] = row
        print(name, row.get("sel", {}).get("lift_top3"), row.get("conf", {}).get("lift_top3"), row.get("final", {}).get("lift_top3"), flush=True)
    (E.OUT / f"rules_{score}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
