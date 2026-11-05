"""
走法マップのコース別の地図（ギアチェンジ・TS・TS持続力・Cr性能・Hペース耐性・基礎スピードの 6 軸で作る地図と、その座標での好走しやすさ）。

ユーザー指示（2026-11-05）: 地図に使う走法軸は、ギアチェンジ・TS・TS持続力・Cr性能・Hペース耐性・基礎スピードの 6 つに固定する（最初は Cr性能までの 4 つ。同日に Hペース耐性、続けて基礎スピードを足した）（MAP_AXES。
  それまでは、コースごとに重要度の高い 3〜6 方面を選んでいた）。この 6 軸で各馬を置き、好走ゾーンも同じ 6 軸の地図の上で算出する。
ユーザー指示（2026-11-03）: 地図は縦軸・横軸の無い「走法の近い馬ほど近くに置く」2 次元にする（任意の軸を座標にしない）。
  6 軸は、丸の色の切り替えと、馬を選んだときの数値バーの強調にも使う。

好走の得点（ユーザー指定）: 1〜3 着 = 1、4 着 0.8、5 着 0.6、6 着 0.4、7 着 0.2、8 着以下 = 0。
期待は「頭数 n のレースで着順がでたらめに決まったときの得点の平均」= Σ_{k=1..n} score(k) / n。

手順（基準を作るときに 1 回だけ。running_map._build_reference から呼ぶ）:
  1. 対象: 2024-01-01 以降の平地（頭数 5 以上）。各走の 9 方面は「その走より前の走」から作ったプロファイル（profile_table）を、
     全馬の分布で順位 → 標準正規にそろえた値（地図と同じ z）。MAP_AXES の 6 軸がすべてある走だけを使う。
  2. 重要度: コースの走で、得点 − 期待 を 6 軸へ線形回帰し、軸ごとに「その軸を抜いたときの R² の落ち」を重要度とする
     （他の軸と重なる分は数えない）。
  3. 地図: 6 軸を、重み w_j = √(6·s_j / Σs)（s_j = 0.5·寄与率 + 0.5·均等。重みの 2 乗の平均が 1）で付けた距離の主成分で 2 次元に置く。
     寄与が小さい軸も配置に効かせるため、寄与率だけでなく均等を半分混ぜる。座標軸に意味は持たせない。
     向きは、好走の得点の 6 軸回帰を地図に写した「好走しやすい向き」が真上。左右は寄与率 2 番目の軸が右に来るように決める。
  4. 好走しやすさ: 地図の 24×24 の格子で、ガウスカーネル（幅 0.5）で近くの走を平滑化し、観測した得点 / 期待 の倍率を出す。
     サンプルの少ないゾーンはコース全体の倍率へ寄せる（M=20）。有意性はレース単位でまとめたサンドウィッチ分散の z と BH 法（FDR 5%）。
  コースの走が MIN_RUNS 未満なら、まず同じ競馬場・芝ダ・距離帯（≤1400 / 〜1700 / 1800〜2599 / 2600+ の長距離）にまとめた「場のグループ」
  （MIN_VENUE_RUNS 以上）、長距離が足りない場は同じ場の 1800m 以上（中長距離）、それも足りなければ競馬場をまたいだ芝ダ・距離帯の「グループ」の地図を使う
  （2026-11-03 ユーザー指示: 長距離など標本の少ない条件は、競馬場はそろえて距離をくくる）。

注意: 重要度・倍率は 2024 年以降の全レースで測った値（標本内）。2024 年の走は位置の方面（Cr性能など）の
実測が少なく（約 13%）、6 軸がそろう走は 2025 年以降が中心になる。
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.research.race import running_map_aptitude as apt

SINCE = "2024-01-01"
MAP_AXES = ("gear", "ts", "sustain", "cr", "endure", "basic")   # 地図に使う走法軸（ユーザー指定 2026-11-05: ギアチェンジ・TS・TS持続力・Cr性能・Hペース耐性・基礎スピード）
# 道悪（稍重以上）の地図: 基礎スピード（速い馬場の軸）の代わりに道悪適性を入れる。芝とダートで別の軸
WET_AXES = {"芝": ("gear", "ts", "sustain", "cr", "endure", "wett"), "ダート": ("gear", "ts", "sustain", "cr", "endure", "wetd")}
WEIGHT_MIX = 0.5     # 地図の距離の重み = 寄与率 × この割合 + 均等 × (1 − この割合)
MIN_MAP_RUNS = 200   # 地図に使う 6 軸がすべてある走がこれ未満のコース（グループ）は地図を作らない
MIN_RUNS = 600
MIN_VENUE_RUNS = 300     # 場のグループ（同じ競馬場の距離帯）の地図を作る最低の走数。背景の倍率はコース全体へ寄せる（M=20）ので、少なめでも極端にはならない
MIN_FIELD = 5


def score_of(pos: np.ndarray) -> np.ndarray:
    """着順 → 好走の得点（1〜3 着 = 1、4〜7 着は 0.2 ずつ下げ、8 着以下 = 0）。"""
    pos = np.asarray(pos, dtype=float)
    return np.where(pos >= 1, np.clip((8 - pos) / 5, 0, 1), 0.0)


def expected_score(n: np.ndarray) -> np.ndarray:
    """頭数 n で着順がでたらめなときの得点の平均。"""
    table = {k: float(score_of(np.arange(1, k + 1)).mean()) for k in range(1, 41)}
    return np.array([table.get(int(v), table[40]) for v in n])


BAND_LABEL = {"短": "1400m以下", "マイル": "1500〜1700m", "中": "1800〜2500m", "長": "2600m以上", "中長": "1800m以上"}


def band_of(distance: Any) -> str:
    d = int(distance)
    return "短" if d <= 1400 else ("マイル" if d < 1800 else ("中" if d < 2600 else "長"))


def venue_group_key(venue: Any, surface: Any, distance: Any) -> str:
    return f"{venue}|{surface}|{band_of(distance)}"


def venue_wide_key(venue: Any, surface: Any, distance: Any) -> Optional[str]:
    """1800m 以上なら、同じ場の 1800m 以上をまとめたキー（長距離が少ない場の受け皿）。それ以外は None。"""
    return f"{venue}|{surface}|中長" if band_of(distance) in ("中", "長") else None


def group_key(surface: Any, distance: Any) -> str:
    return f"{surface}|{band_of(distance)}"


def _importance(Z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """r を Z へ線形回帰。方面ごとの R² の落ち・係数・全体の R²。"""
    X = np.column_stack([np.ones(len(Z)), Z])
    tss = float(((r - r.mean()) ** 2).sum()) or 1e-12

    def r2(cols: list[int]) -> tuple[float, np.ndarray]:
        Xs = X[:, [0] + [c + 1 for c in cols]]
        beta, *_ = np.linalg.lstsq(Xs, r, rcond=None)
        return 1 - float(((r - Xs @ beta) ** 2).sum()) / tss, beta

    k = Z.shape[1]
    full, beta = r2(list(range(k)))
    drop = np.array([full - r2([c for c in range(k) if c != j])[0] for j in range(k)])
    return np.maximum(drop, 0), beta[1:], full


def _orient(Y: np.ndarray, A: np.ndarray, up: np.ndarray, right_row: int) -> tuple[np.ndarray, np.ndarray]:
    """地図の向きを決める。up（2 次元）が真上、A の right_row 行（その方面の向き）が右側になるように回転・反転する。"""
    if np.linalg.norm(up) > 1e-9:
        rot = math.pi / 2 - math.atan2(up[1], up[0])
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        Y, A = Y @ R.T, A @ R.T
    if A[right_row][0] < 0:
        Y, A = Y * [-1, 1], A * [-1, 1]
    return Y, A


def _dist_rho(X: np.ndarray, Y: np.ndarray, n: int = 400, seed: int = 0) -> float:
    """重み付き 6 軸の距離と地図上の距離の順位相関（標本 n 頭の全組）。"""
    from scipy.stats import spearmanr

    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    iu = np.triu_indices(len(idx), 1)
    dx = np.linalg.norm(X[idx][:, None] - X[idx][None], axis=2)[iu]
    dy = np.linalg.norm(Y[idx][:, None] - Y[idx][None], axis=2)[iu]
    return float(spearmanr(dx, dy).statistic)


def weights_of(share: np.ndarray, order: list[int]) -> np.ndarray:
    """地図の距離の重み。上位の方面（order）だけ w_j = √(K·share_j / Σ上位の share)（上位の重みの 2 乗の平均が 1）、ほかは 0。"""
    w = np.zeros(len(share))
    top = share[order]
    tot = top.sum()
    w[order] = np.sqrt(len(order) * (top / tot if tot > 0 else np.full(len(order), 1.0 / len(order))))
    return w


def place(Z: np.ndarray, cm: dict[str, Any]) -> np.ndarray:
    """9 方面の z（欠けは 0 = 全馬の中央値）を、コースの地図の座標へ。"""
    return (np.nan_to_num(Z) * np.array(cm["w"]) - np.array(cm["mu"])) @ np.array(cm["A"])


def _heat(P: np.ndarray, y: np.ndarray, e: np.ndarray, race_codes: np.ndarray, extent: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import norm

    grid = apt._grid(extent)
    co, cx = apt._obs_exp(grid, P, y, e)
    base = float(y.sum() / max(e.sum(), 1e-9))
    valid = cx > apt.MIN_EXPECTED
    z = apt._cluster_z(grid, P, y - e * base, race_codes)
    sel = np.zeros(len(grid), bool)
    if valid.any():
        sel[np.where(valid)[0][apt._bh(2 * norm.sf(np.abs(z[valid])))]] = True
    lift = (co + apt.SHRINK * base) / (cx + apt.SHRINK) / base   # コース全体を 1 とした倍率
    sig = np.where(~valid, 0, np.where(sel & (z > 0), 2, 1))
    return {"lift": [round(float(v), 3) for v in lift], "sig": "".join(str(int(v)) for v in sig), "base": round(base, 3)}


def _axes_for(axes: Any, surface: Any) -> tuple[str, ...]:
    """地図に使う軸。dict なら芝ダごとに別（道悪の地図は芝=芝道悪適性・ダート=ダート道悪適性）。"""
    return tuple(axes[surface]) if isinstance(axes, dict) else tuple(axes)


def _one(g: pd.DataFrame, Zall: np.ndarray, present: np.ndarray, axis_keys: list[str], axes: Any = MAP_AXES) -> Optional[dict[str, Any]]:
    idx = g.index.to_numpy()
    Z, pr = Zall[idx], present[idx]
    r = g["y"].to_numpy() - g["e"].to_numpy()
    fixed = [axis_keys.index(k) for k in _axes_for(axes, g["surface"].iloc[0])]
    ok = pr[:, fixed].all(axis=1)
    if ok.sum() < MIN_MAP_RUNS:
        return None
    # 重要度は 6 軸だけの回帰で測る（他の軸は入れない）。大きい順に並べ、寄与の割合を重みにする
    drop4, beta4, full = _importance(Z[ok][:, fixed], r[ok])
    pos = np.argsort(-drop4)
    order = [fixed[i] for i in pos]
    keys = [axis_keys[j] for j in order]
    drop, beta = np.zeros(len(axis_keys)), np.zeros(len(axis_keys))
    drop[fixed], beta[fixed] = drop4, beta4
    share = drop / drop.sum() if drop.sum() > 0 else np.zeros(len(drop))
    if drop.sum() <= 0:
        share[fixed] = 1.0 / len(fixed)
    # 地図の距離の重みは、寄与率と均等の半々（寄与が小さい軸も配置に効かせる）
    w = weights_of(WEIGHT_MIX * share + (1 - WEIGHT_MIX) * np.isin(np.arange(len(share)), fixed) / len(fixed), order)
    Zw = Z[ok] * w
    mu = Zw.mean(0)
    Zc = Zw - mu
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    A0 = Vt[:2].T
    # 好走の得点の 6 軸回帰を地図に写した向きを「上」にする（得点 ≈ b·z = (b/w)·zw）
    bw = np.zeros(len(w))
    bw[order] = beta4[pos] / w[order]
    Y, A = _orient(Zc @ A0, A0, A0.T @ bw, order[1])
    recon = Y @ A.T
    var = Zc.var(0)
    extent = apt.extent_from(Y)
    sub = g[ok]
    heat = _heat(Y, sub["y"].to_numpy(), sub["e"].to_numpy(), pd.factorize(sub["race_id"])[0], extent)
    return {
        "axes": keys,
        "keys": list(axis_keys),
        "w": [round(float(v), 4) for v in w],
        "distance_spearman": round(_dist_rho(Zc, Y), 3),
        "importance": [{"key": axis_keys[j], "share": round(float(share[j]), 3), "r2_drop": round(float(drop[j]), 5),
                        "sign": int(np.sign(beta[j])) if beta[j] != 0 else 0} for j in order],
        "r2_full": round(float(full), 4),
        "mu": mu.tolist(), "A": A.tolist(),
        "r2": [None if v < 1e-12 else round(float(1 - (Zc[:, j] - recon[:, j]).var() / v), 3) for j, v in enumerate(var)],
        "top_k": len(order),
        "top_share": 1.0,
        "explained": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 3),
        "extent": extent,
        "n_runs": int(len(sub)), "n_races": int(sub["race_id"].nunique()),
        **heat,
    }


def build(tab: pd.DataFrame, Z: np.ndarray, present: np.ndarray, axis_keys: list[str], axes: Any = MAP_AXES, mask: Optional[np.ndarray] = None) -> dict[str, Any]:
    """tab: race_id, date, venue, surface, distance, finish_position, n_field（Z・present と同じ並び）。
    Z: 9 方面の z（地図と同じ変換・符号。欠けは NaN）。present: 実測由来の値があるか。
    axes: 地図に使う軸（dict なら芝ダごと）。mask: 地図に使う走（良馬場だけ・道悪だけ、など。省略時は全部）。"""
    if mask is not None:
        tab, Z, present = tab[mask], Z[mask], present[mask]
    t = tab.reset_index(drop=True).copy()
    t["y"] = score_of(t["finish_position"].to_numpy())
    t["e"] = expected_score(t["n_field"].to_numpy())
    keep = ((t["date"] >= pd.Timestamp(SINCE)) & (t["n_field"] >= MIN_FIELD) & t[["venue", "surface", "distance"]].notna().all(axis=1)
            & t["finish_position"].notna()).to_numpy()
    t = t[keep].reset_index(drop=True)
    Zk = np.nan_to_num(Z[keep])
    pk = present[keep]

    courses: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    vgroups: dict[str, Any] = {}
    t["ckey"] = [apt.course_key(v, s, d) for v, s, d in zip(t["venue"], t["surface"], t["distance"])]
    t["gkey"] = [group_key(s, d) for s, d in zip(t["surface"], t["distance"])]
    t["vkey"] = [venue_group_key(v, s, d) for v, s, d in zip(t["venue"], t["surface"], t["distance"])]
    for gk, g in t.groupby("gkey"):
        if len(g) >= MIN_RUNS:
            res = _one(g, Zk, pk, axis_keys, axes)
            if res:
                s_, b_ = gk.split("|")
                groups[gk] = {**res, "level": "group", "label": f"{s_} {BAND_LABEL[b_]}（全場）"}
    for ck, g in t.groupby("ckey"):
        if len(g) >= MIN_RUNS:
            res = _one(g, Zk, pk, axis_keys, axes)
            if res:
                courses[ck] = {**res, "level": "course"}
    # 場のグループ: 地図の無いコースがある集まりだけ（地図のあるコースだけなら使われない）
    t["wkey"] = [venue_wide_key(v, s, d) for v, s, d in zip(t["venue"], t["surface"], t["distance"])]
    for col in ("vkey", "wkey"):
        for vk, g in t[t[col].notna()].groupby(col):
            if len(g) < MIN_VENUE_RUNS or set(g["ckey"]) <= set(courses):
                continue
            res = _one(g, Zk, pk, axis_keys, axes)
            if res:
                v_, s_, b_ = vk.split("|")
                vgroups[vk] = {**res, "level": "venue_group", "label": f"{v_} {s_} {BAND_LABEL[b_]}"}
    return {"since": SINCE, "score": "1〜3着=1, 4着0.8, 5着0.6, 6着0.4, 7着0.2, 8着以下=0", "courses": courses, "venue_groups": vgroups, "groups": groups}


def lookup(maps: dict[str, Any], venue: Any, surface: Any, distance: Any) -> Optional[dict[str, Any]]:
    """コースの地図。無ければ同じ競馬場・芝ダ・距離帯の地図、それも無ければ芝ダ・距離帯（全場）の地図。どれも無ければ None。"""
    if not (maps and venue and surface and distance):
        return None
    hit = maps.get("courses", {}).get(apt.course_key(venue, surface, distance))
    if hit:
        return hit
    vg = maps.get("venue_groups", {})
    hit = vg.get(venue_group_key(venue, surface, distance)) or vg.get(venue_wide_key(venue, surface, distance) or "")
    if hit:
        return hit
    return maps.get("groups", {}).get(group_key(surface, distance))
