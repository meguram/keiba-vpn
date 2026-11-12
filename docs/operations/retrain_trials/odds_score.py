"""オッズ予測の合否（§5.1）: 2025 年の log オッズ RMSE 単勝 ≤0.464・複勝 ≤0.385、かつ年別（2022〜2025）の最大が全体の 1.1 倍以内。
    python docs/operations/retrain_trials/odds_score.py <出力先>...   （wp_oof_pred.parquet の win_B・place_B）"""
import sys
import numpy as np, pandas as pd
for d in sys.argv[1:]:
    p = pd.read_parquet(f"{d}/wp_oof_pred.parquet")
    row = [d.rsplit("/", 1)[-1]]
    for c, a, line in (("win_B", "win_odds", 0.464), ("place_B", "place_mid", 0.385)):
        e = (np.log(p[c]) - np.log(p[a])) ** 2
        ok = e.notna()
        allr = float(np.sqrt(e[ok].mean())); by = {int(y): float(np.sqrt(v.mean())) for y, v in e[ok].groupby(p.loc[ok, "year"])}
        r25 = by[2025]; mx = max(by.values())
        row.append(f"{c}: 2025 {r25:.4f}{'○' if r25 <= line else '×'} 全体 {allr:.4f} 年別最大/全体 {mx/allr:.3f}{'○' if mx <= 1.1*allr else '×'} 年別 " + " ".join(f"{y}:{v:.3f}" for y, v in by.items()))
    print(" | ".join(row))
