"""血統メタクラスタ L2 を「得意領域 (相対 lift)」軸で再構築する。

現状の `unified.parquet` の L2 は「絶対勝率 - 全体平均」を特徴量とした KMeans の結果で、
「全条件で勝率が高い → L2=4」のように **絶対水準** に支配されている。

本スクリプトでは、各種牡馬の条件別勝率を **自分の全体勝率に対する倍率 (lift)** に正規化してから
KMeans を回し、「**得意領域パターンを共有するグループ**」として L2 を再定義する。

入出力:
    input : data/research/bloodline_meta_cluster/unified.parquet (L2 列を上書き)
            data/research/bloodline_meta_cluster/scaler.json (新規生成)
    output: data/research/bloodline_meta_cluster/unified.parquet (in-place 更新)
            data/research/bloodline_meta_cluster/l2_profiles.json (上書き)
            data/research/bloodline_meta_cluster/scaler_strength.json (新規)

Usage:
    python -m src.research.pedigree.rebuild_meta_cluster_by_strength [--k K_FIX]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
ART_DIR = ROOT / "data/page_reference/note_aptitude_race"

# 条件特徴量 (絶対勝率) のうち、相対化対象にするカラム
COND_COLS_BASE = [
    "win_v_東京", "win_v_中山", "win_v_阪神", "win_v_京都", "win_v_中京",
    "win_v_新潟", "win_v_小倉", "win_v_福島", "win_v_札幌", "win_v_函館",
    "win_s_芝", "win_s_ダート",
    "win_d_短距離", "win_d_マイル", "win_d_中距離", "win_d_長距離",
    "win_pace_スロー瞬発力_短距離", "win_pace_スロー瞬発力_マイル",
    "win_pace_スロー瞬発力_中距離", "win_pace_スロー瞬発力_長距離",
    "win_pace_持続力勝負_短距離",   "win_pace_持続力勝負_マイル",
    "win_pace_持続力勝負_中距離",   "win_pace_持続力勝負_長距離",
    "win_pace_持久力勝負_短距離",   "win_pace_持久力勝負_マイル",
    "win_pace_持久力勝負_中距離",   "win_pace_持久力勝負_長距離",
    "win_steep", "win_flat", "win_heavy",
]


def _relativize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """各条件勝率を自分の win_eb_total で割り「lift」(1.0 = 自分の平均的水準) に変換。

    NaN は 1.0 で埋める (= 自分の平均と同等)。
    上限/下限はクリップ (極端値防止)。
    """
    out = pd.DataFrame(index=df.index)
    base = df["win_eb_total"].clip(lower=0.001)
    for c in cols:
        v = df[c] / base
        out[c] = v.fillna(1.0).clip(lower=0.3, upper=3.0)
    return out


def _select_k(X: np.ndarray, k_range=range(4, 11), random_state: int = 42) -> tuple[int, dict]:
    """シルエットスコアで最適 k を選定。"""
    sil_scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sc = silhouette_score(X, labels)
        sil_scores[k] = sc
    best_k = max(sil_scores.items(), key=lambda x: x[1])[0]
    return best_k, sil_scores


def build(k_fix: int | None = None) -> dict:
    print(f"loading {ART_DIR/'unified.parquet'}", flush=True)
    uni = pd.read_parquet(ART_DIR / "unified.parquet")
    print(f"  shape={uni.shape}, L2 旧分布: {dict(uni['L2'].value_counts().sort_index())}", flush=True)

    # 主流 stallion のみクラスタリング対象
    mask_main = (uni["entity_type"] == "stallion") & (uni["L2"] >= 0)
    sub = uni[mask_main].copy()

    rel = _relativize(sub, COND_COLS_BASE)
    print(f"  features after relativize: {rel.shape}", flush=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(rel.values)

    # k 選定
    if k_fix is not None:
        best_k, sil_scores = k_fix, {}
    else:
        best_k, sil_scores = _select_k(X)
    print(f"  best_k={best_k}, silhouette_scores={sil_scores}", flush=True)

    # 最終クラスタリング
    km = KMeans(n_clusters=best_k, random_state=42, n_init=50)
    labels = km.fit_predict(X)

    # ── 外れ値クラスタ (size < MIN_SIZE) を最近接の有効クラスタへ吸収 ──
    MIN_SIZE = 5
    counts = pd.Series(labels).value_counts().to_dict()
    bad = {c for c, n in counts.items() if n < MIN_SIZE}
    good = [c for c in counts if c not in bad]
    if bad and good:
        good_centroids = {c: km.cluster_centers_[c] for c in good}
        for i, lbl in enumerate(labels):
            if lbl in bad:
                # 最近接の有効クラスタへ
                d_min, best = float("inf"), good[0]
                for c, cent in good_centroids.items():
                    d = np.linalg.norm(X[i] - cent)
                    if d < d_min:
                        d_min, best = d, c
                labels[i] = best

    # ラベルを 0..K-1 に詰め直す
    uniq = sorted(set(labels))
    relabel = {old: new for new, old in enumerate(uniq)}
    labels = np.array([relabel[l] for l in labels])

    sub["L2_new"] = labels
    print(f"  new L2 distribution: {dict(pd.Series(labels).value_counts().sort_index())}", flush=True)

    # unified に書き戻し
    uni.loc[sub.index, "L2"] = sub["L2_new"].astype(int)
    # 主流 stallion 以外は -1 で残す
    uni.loc[~mask_main, "L2"] = -1

    # プロファイル (各クラスタの平均 lift) を作成
    rel_with_l2 = rel.copy()
    rel_with_l2["L2"] = labels
    profiles = {}
    for L2 in sorted(set(labels)):
        means = rel_with_l2[rel_with_l2["L2"] == L2][COND_COLS_BASE].mean()
        # 「lift - 1」に変換 (0 = 平均的、+ = 得意、- = 苦手)
        profiles[int(L2)] = {c: round(float(means[c] - 1.0), 4) for c in COND_COLS_BASE}

    # 保存
    (ART_DIR / "unified.parquet").unlink()
    uni.to_parquet(ART_DIR / "unified.parquet")
    (ART_DIR / "l2_profiles.json").write_text(
        json.dumps({str(k): v for k, v in profiles.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # scaler 情報も新規 (KNN 用と別ファイル) に保存
    (ART_DIR / "scaler_strength.json").write_text(json.dumps({
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": COND_COLS_BASE,
        "n_clusters": int(best_k),
        "silhouette_scores": sil_scores,
        "method": "lift_normalized_kmeans",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: unified.parquet, l2_profiles.json, scaler_strength.json", flush=True)
    return {"best_k": int(best_k), "n_main": int(mask_main.sum()), "silhouette_scores": sil_scores}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=None, help="クラスタ数を固定 (省略時はシルエットで自動選定)")
    args = p.parse_args()
    res = build(k_fix=args.k)
    print(json.dumps(res, ensure_ascii=False, indent=2))
