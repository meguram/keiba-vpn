"""tp-84: 予想したペース（前半5F・RPCI のずれ）× 馬のタイプ・感応度 → 『この流れが向きそうか』の特徴量。as-of（今回の日付より前の走だけ）。

事前調査（tp84_pace_aptitude）: 後傾なほど先行馬が、前傾なほど後ろの馬が着順の位置で 0.03〜0.07 上振れする。前期・後期で傾きが揃った。
  PF_A  レース全体の予想（pp_*）＋ 出走馬内での相対の脚質・上がり順位（モデルが自分で組み合わせる材料）
  PF_B  明示の交互作用: 脚質・上がりの相対位置 × 予想ペース / 馬ごとの感応度（過去走の上振れを実際のペースに回帰した傾き）× 予想ペース /
        予想ペースに近かった過去走での上振れの平均（カーネル平均）
上振れ r_fp = −(着順の位置 − 過去5走の期待)。ペースは par からのずれを、2021 年以前の標準偏差で割った値。
先読みを避ける約束: 過去走は 1 走以上ずらす。今回のレースの結果・実際のペースは使わない（予想ペースだけ）。市場オッズ・人気は使わない。キーは race_id / horse_id。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs
from src.pipeline.features import tp42_racelevel as rl

KEY = ["race_id", "horse_id"]
L = 10                 # 感応度に使う過去走数
DECAY = 0.85           # 1 走古くなるごとの重み
LAM_SLOPE = 3.0        # 傾きを 0 へ寄せる強さ（擬似的な走数）
LAM_KERN = 1.5         # カーネル平均を 0 へ寄せる強さ
BW = 1.0               # カーネルの幅（標準化したペース単位）
PP = ["pp_F5", "pp_RPCI", "pp_HYB", "pp_L3", "pp_bal"]


def _rel(d: pd.DataFrame, col: str) -> np.ndarray:
    """レース内の順位の割合 − 0.5（出走馬の過去走の値どうしの比較。値のない馬は NaN）。"""
    return (d.groupby("race_id")[col].rank(pct=True) - 0.5).to_numpy(float)


def _slope_and_kernel(R: np.ndarray, X: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R: (n, L) 過去走の上振れ、X: (n, L) 過去走の実際のペース（標準化）、P: (n,) 今回の予想ペース（標準化）。"""
    w = DECAY ** np.arange(R.shape[1])[None, :]
    ok = np.isfinite(R) & np.isfinite(X)
    Rz, Xz = np.where(ok, R, 0.0), np.where(ok, X, 0.0)
    wm = w * ok
    b = (wm * Xz * Rz).sum(1) / ((wm * Xz ** 2).sum(1) + LAM_SLOPE)
    with np.errstate(invalid="ignore"):
        k = np.exp(-((Xz - np.nan_to_num(P)[:, None]) ** 2) / (2 * BW ** 2))
    wk = wm * k
    kern = (wk * Rz).sum(1) / (wk.sum(1) + LAM_KERN)
    gen = (wm * Rz).sum(1) / (wm.sum(1) + LAM_KERN)
    return b, kern, kern - gen


def build(d: pd.DataFrame, pace: pd.DataFrame, real: pd.DataFrame, sd: dict[str, float]) -> dict[str, pd.DataFrame]:
    """d: fs.prep 済みで horse_id, dnum, race_id の順に並んだ走表。pace: race_id と PP 列（予想）。real: race_id と d_F5, d_RPCI（実際のずれ）。sd: {'F5': .., 'RPCI': ..}。"""
    d = d.copy()
    d = d.merge(pace[["race_id"] + PP], on="race_id", how="left").merge(real[["race_id", "d_F5", "d_RPCI"]], on="race_id", how="left")
    exp_fp = rl._ewm(fs.lags(d, "fp", 5))
    d["r_fp"] = -(d["fp"].to_numpy(float) - exp_fp)
    d["x_F5"], d["x_RPCI"] = d["d_F5"] / sd["F5"], d["d_RPCI"] / sd["RPCI"]
    R = fs.lags(d, "r_fp", L)
    d["sty"] = np.nanmean(fs.lags(d, "pos_first", 3), axis=1)
    d["kick"] = np.nanmean(fs.lags(d, "l3f_rank", 3), axis=1)
    sty, kick = _rel(d, "sty"), _rel(d, "kick")
    P = {"F5": d["pp_F5"].to_numpy(float) / sd["F5"], "RPCI": d["pp_RPCI"].to_numpy(float) / sd["RPCI"]}
    b = pd.DataFrame(index=d.index)
    b["sty_rel"], b["kick_rel"] = sty, kick
    for t in ("RPCI", "F5"):
        b[f"sty_x_{t}"], b[f"kick_x_{t}"] = sty * P[t], kick * P[t]
        slope, kern, kcon = _slope_and_kernel(R, fs.lags(d, f"x_{t}", L), P[t])
        b[f"sens_{t}"], b[f"kern_{t}"], b[f"kernc_{t}"] = slope * P[t], kern, kcon
    A = pd.concat([d[PP], b[["sty_rel", "kick_rel"]]], axis=1)
    B = b.drop(columns=["sty_rel", "kick_rel"])
    out = {"PF_A": A, "PF_B": B, "PF_ALL": pd.concat([A, B], axis=1)}
    return {k: pd.concat([d[KEY].reset_index(drop=True), v.reset_index(drop=True)], axis=1) for k, v in out.items()}
