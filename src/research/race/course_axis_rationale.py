"""コース別の走法軸の優位を言語化するための判定（純粋な計算。データの測定は src/scripts/data/course_axis_rationale.py）。

考え方: コース単位の差は偶然のぶれと区別できない（乱数ラベルでも同じ数だけ出る）ので、言語化の単位は『コースの形状』にする。
形状（直線長・小回り・ダート・坂・距離）と軸の関係を、2024-25 年（fit）と 2026 年（確認のみ）の両方で、同じ向きに |t|>=T_REPL で出たものだけ『根拠あり』にする。
根拠ありの形状効果だけを、そのコースの形状の値から日本語の文にする。人気は使わない。
"""
from __future__ import annotations

import math

import numpy as np

T_REPL = 1.9          # 両期間で同じ向きに、これ以上の |t|
T_COURSE = 2.3        # コース単位の目印
FEATURES = ["直線長", "小回り", "ダート", "坂", "距離"]
FEATURE_UNIT = {"直線長": "100m", "小回り": "あり/なし", "ダート": "ダート/芝", "坂": "あり/なし", "距離": "400m"}
AXIS_NAME = {"gear": "ギアチェンジ", "ts": "TS（末脚の速さ）", "sustain": "TS持続力", "cr": "Cr性能", "endure": "Hペース耐性", "basic": "基礎スピード"}


def shape_features(items: list[dict], surface: str, distance: int) -> list[float] | None:
    """コースマスタの項目から、切片つきの形状ベクトル [1, 直線長, 小回り, ダート, 坂, 距離] を返す。内回り・外回りで直線長が違うコースは None（混ぜない）。"""
    st = sorted({x["final_straight_m"] for x in items if x.get("final_straight_m")})
    if not st or st[-1] - st[0] > 1:
        return None
    small = float(any((x.get("corner_radius_class_A") or "").startswith("小回り") for x in items))
    slope = float(any(x.get("goal_slope_exists") for x in items))
    return [1.0, (st[0] - 400) / 100, small, 1.0 if surface == "ダート" else 0.0, slope, (distance - 1600) / 400]


def wls(X: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """重み付き回帰。残差のばらつきが標準誤差より大きいときは標準誤差を広げる。"""
    w = 1 / se ** 2
    W = X.T @ (X * w[:, None])
    b = np.linalg.solve(W, X.T @ (w * y))
    res = y - X @ b
    infl = max(float((w * res ** 2).sum() / max(len(y) - X.shape[1], 1)), 1.0)
    return b, np.sqrt(np.diag(np.linalg.inv(W)) * infl)


def collect(stats: dict, period: str, feats: dict[str, list[float]], axes: list[str]):
    """stats[コース][期間] = {b, se}。形状が取れるコースだけを (キー, X, Y, S) にする。"""
    keys, X, Y, S = [], [], [], []
    for ck, row in stats.items():
        r = row.get(period)
        f = feats.get(ck)
        if r and f:
            keys.append(ck)
            X.append(f)
            Y.append(r["b"])
            S.append(r["se"])
    return keys, np.array(X), np.array(Y), np.array(S)


def shape_effects(stats: dict, feats: dict[str, list[float]], axes: list[str], fit: str = "fit", check: str = "y26") -> list[dict]:
    """(形状, 軸) ごとの係数を fit と確認で出し、両方で同じ向き・|t|>=T_REPL なら verdict='根拠あり'。"""
    k1, X1, Y1, S1 = collect(stats, fit, feats, axes)
    k2, X2, Y2, S2 = collect(stats, check, feats, axes)
    out = []
    for j, ax in enumerate(axes):
        b1, s1 = wls(X1, Y1[:, j], S1[:, j])
        b2, s2 = wls(X2, Y2[:, j], S2[:, j])
        for i, f in enumerate(FEATURES, start=1):
            t1, t2 = float(b1[i] / s1[i]), float(b2[i] / s2[i])
            same = np.sign(t1) == np.sign(t2)
            if same and abs(t1) >= T_REPL and abs(t2) >= T_REPL:
                v = "根拠あり"
            elif abs(t1) >= T_REPL or abs(t2) >= T_REPL:
                v = "片方の期間のみ"
            else:
                v = "差なし"
            out.append({"axis": ax, "feature": f, "b_fit": float(b1[i]), "t_fit": t1, "b_chk": float(b2[i]), "t_chk": t2,
                        "same_sign": bool(same), "verdict": v, "n_fit": len(k1), "n_chk": len(k2)})
    return out


def bh(p: list[float]) -> list[float]:
    """Benjamini-Hochberg の q 値。"""
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def p_from_t(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def course_counts(stats: dict, period: str, axes: list[str], t_mark: float = T_COURSE) -> dict:
    """コース×軸の検定の数。|t|>=t_mark の個数と、全体の個数、t の標準偏差（偶然だけなら 1 付近）。"""
    t = [r[period]["b"][j] / r[period]["se"][j] for r in stats.values() if r.get(period) for j in range(len(axes))]
    arr = np.array(t)
    return {"n": int(len(arr)), "hits": int((np.abs(arr) >= t_mark).sum()), "sd": float(arr.std()) if len(arr) else float("nan")}


def predicted_shift(coef: list[float], feat: list[float], i: int) -> float:
    """形状 i（1 始まり）が基準（直線 400m・大回り・芝・坂なし・1600m）からずれている分による、係数の予測ずれ。"""
    return float(coef[i] * feat[i])


def sentence(axis: str, feature: str, sign: int, feat_value: str) -> str:
    """根拠あり（形状, 軸）から作る、そのコース向けの文。向きは推定値の符号（sign>0 で『効きやすい』）。"""
    way = "効きやすい（優位寄り）" if sign > 0 else "効きにくい（不利寄り）"
    return f"{feat_value}なので、{AXIS_NAME[axis]}が{way}"
