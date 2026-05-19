"""L2 (適性領域) を「得意領域」観点で **細粒度化** してクラスタを再構築する。

旧構成:
    L2 (4 クラスタ) → L3 (視点別、L2 内サブクラスタ)

新構成 (本スクリプトの出力):
    L2_fine (10〜18 クラスタ) — 得意領域を細かく区別 (例: 「東京芝マイル瞬発型」「中山ダート短距離型」)
    SuperGroup (4〜6 個)    — L2_fine centroid の階層クラスタリングで得る大カテゴリ
    + 2D 座標 (PCA + UMAP) — 多次元空間での「位置」を可視化用に保持

特徴量:
    各種牡馬の条件別勝率を ``win_eb_total`` で割った **相対 lift**
    (1.0 = 自分の平均水準, >1 = その条件で得意, <1 = 苦手) を StandardScaler で正規化。

入出力:
    input :
        data/research/bloodline_meta_cluster/unified.parquet
    output:
        data/research/bloodline_meta_cluster/unified.parquet         (L2 列を細粒度に上書き)
        data/research/bloodline_meta_cluster/l2_profiles.json        (各 L2 の平均 lift)
        data/research/bloodline_meta_cluster/l2_centroids.json       (各 L2 の centroid ベクトル, 標準化後)
        data/research/bloodline_meta_cluster/l2_positions_2d.json    (PCA / UMAP 2D 座標)
        data/research/bloodline_meta_cluster/l2_super_groups.json    (L2 → super_group ID マッピング)
        data/research/bloodline_meta_cluster/l2_similarity.json      (クラスタ間距離 + 近接ペア)
        data/research/bloodline_meta_cluster/scaler_strength.json    (StandardScaler パラメータ等)

Usage:
    python -m src.research.pedigree.build_l2_fine_clusters [--k_min 10 --k_max 18 --n_super 5]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
ART_DIR = ROOT / "data/research/bloodline_meta_cluster"

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
    out = pd.DataFrame(index=df.index)
    base = df["win_eb_total"].clip(lower=0.001)
    for c in cols:
        v = df[c] / base
        out[c] = v.fillna(1.0).clip(lower=0.3, upper=3.0)
    return out


def _select_k(
    X: np.ndarray, k_range: range, method: str = "agglomerative", random_state: int = 42
) -> tuple[int, dict[int, float]]:
    """シルエットスコアで最適 k を選定。"""
    sil_scores: dict[int, float] = {}
    for k in k_range:
        if method == "kmeans":
            algo = KMeans(n_clusters=k, random_state=random_state, n_init=20)
            labels = algo.fit_predict(X)
        else:
            algo = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = algo.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sc = silhouette_score(X, labels)
        sil_scores[k] = float(sc)
    best_k = max(sil_scores.items(), key=lambda x: x[1])[0]
    return best_k, sil_scores


def _merge_small_clusters(
    labels: np.ndarray, X: np.ndarray, min_size: int = 5
) -> np.ndarray:
    """小さすぎるクラスタ (n<min_size) を最近接の有効クラスタへ吸収。"""
    counts = pd.Series(labels).value_counts().to_dict()
    bad = {c for c, n in counts.items() if n < min_size}
    good = [c for c in counts if c not in bad]
    if not bad or not good:
        return labels
    centroids = {c: X[labels == c].mean(axis=0) for c in good}
    new_labels = labels.copy()
    for i, lbl in enumerate(labels):
        if lbl in bad:
            d_min, best = float("inf"), good[0]
            for c, cent in centroids.items():
                d = float(np.linalg.norm(X[i] - cent))
                if d < d_min:
                    d_min, best = d, c
            new_labels[i] = best
    # 0..K-1 に詰め直す
    uniq = sorted(set(new_labels))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[c] for c in new_labels], dtype=int)


def _compute_2d_positions(
    centroids: np.ndarray, X_stallions: np.ndarray, stallion_labels: np.ndarray
) -> dict[str, Any]:
    """centroid + 種牡馬の 2D 座標を PCA / UMAP で計算。"""
    res: dict[str, Any] = {}
    # PCA: centroid と種牡馬を同じ空間に投影
    all_pts = np.vstack([centroids, X_stallions])
    pca = PCA(n_components=2, random_state=42)
    proj_pca = pca.fit_transform(all_pts)
    n_c = centroids.shape[0]
    res["pca"] = {
        "centroids": proj_pca[:n_c].tolist(),
        "stallions": proj_pca[n_c:].tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    # UMAP: 種牡馬を埋め込んだ後、centroid を transform
    try:
        import umap  # type: ignore

        # n_neighbors は データサイズ依存 (種牡馬数 ~130)
        n_neighbors = min(15, max(5, X_stallions.shape[0] // 10))
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
            metric="cosine", random_state=42,
        )
        umap_stallions = reducer.fit_transform(X_stallions)
        umap_centroids = reducer.transform(centroids)
        res["umap"] = {
            "centroids": umap_centroids.tolist(),
            "stallions": umap_stallions.tolist(),
            "params": {"n_neighbors": n_neighbors, "min_dist": 0.3, "metric": "cosine"},
        }
    except Exception as e:
        res["umap"] = {"error": str(e)}
    return res


def _build_super_groups(
    centroids: np.ndarray, n_super: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """centroid をコサイン距離で階層クラスタリングし super-group を構築。"""
    if centroids.shape[0] <= n_super:
        return np.arange(centroids.shape[0]), {"note": "n_clusters <= n_super, identity mapping"}
    dist = cosine_distances(centroids)
    # precomputed 距離で AgglomerativeClustering
    agg = AgglomerativeClustering(
        n_clusters=n_super, metric="precomputed", linkage="average",
    )
    sg_labels = agg.fit_predict(dist)
    return sg_labels.astype(int), {
        "method": "agglomerative_cosine_average",
        "n_super": int(n_super),
        "input_n_l2": int(centroids.shape[0]),
    }


def _compute_similarity(centroids: np.ndarray) -> dict[str, Any]:
    """centroid 間のコサイン類似度と最近接ペアを計算。"""
    cos_sim = 1.0 - cosine_distances(centroids)  # [-1, 1]
    n = centroids.shape[0]
    # 各クラスタの最近接 top-3 (自分を除く)
    nearest: dict[int, list[dict[str, Any]]] = {}
    farthest: dict[int, list[dict[str, Any]]] = {}
    for i in range(n):
        sims = [(j, float(cos_sim[i, j])) for j in range(n) if j != i]
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        nearest[i] = [{"L2": j, "cos_sim": s} for j, s in sims_sorted[:3]]
        farthest[i] = [{"L2": j, "cos_sim": s} for j, s in sims_sorted[-3:]]
    return {
        "cos_sim_matrix": cos_sim.round(4).tolist(),
        "euclid_dist_matrix": euclidean_distances(centroids).round(4).tolist(),
        "nearest_top3": nearest,
        "farthest_top3": farthest,
    }


def build(
    k_min: int = 10, k_max: int = 18, n_super: int = 5,
    method: str = "agglomerative", k_fix: int | None = None,
) -> dict[str, Any]:
    print(f"[load] {ART_DIR/'unified.parquet'}", flush=True)
    uni = pd.read_parquet(ART_DIR / "unified.parquet")
    print(f"  shape={uni.shape}, L2(旧) 分布: {dict(uni['L2'].value_counts().sort_index())}", flush=True)

    # 主流 stallion のみクラスタリング対象
    mask_main = (uni["entity_type"] == "stallion") & (uni["L2"] >= 0)
    # まだ L2 が再構築前なら entity_type=='stallion' 全部を対象
    if mask_main.sum() == 0:
        mask_main = uni["entity_type"] == "stallion"
    sub = uni[mask_main].copy()

    rel = _relativize(sub, COND_COLS_BASE)
    print(f"  features after relativize: {rel.shape}", flush=True)

    scaler = StandardScaler()
    X_full = scaler.fit_transform(rel.values)

    # PCA で次元圧縮 (Curse of dimensionality 軽減) → クラスタリングが安定
    pca_reducer = PCA(n_components=min(8, X_full.shape[1] - 1), random_state=42)
    X = pca_reducer.fit_transform(X_full)
    cum_var = float(pca_reducer.explained_variance_ratio_.sum())
    print(f"  PCA reduce: {X_full.shape[1]}D → {X.shape[1]}D, "
          f"cumulative variance={cum_var:.3f}", flush=True)

    # k 選定 (圧縮後の空間でシルエット)
    if k_fix is not None:
        best_k, sil_scores = k_fix, {}
    else:
        best_k, sil_scores = _select_k(X, range(k_min, k_max + 1), method=method)
    print(f"  best_k={best_k}, silhouette_scores={sil_scores}", flush=True)

    # 最終クラスタリング
    if method == "kmeans":
        algo = KMeans(n_clusters=best_k, random_state=42, n_init=50)
        labels = algo.fit_predict(X)
    else:
        algo = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        labels = algo.fit_predict(X)

    # 小クラスタ吸収
    labels = _merge_small_clusters(labels, X, min_size=4)
    sub["L2_new"] = labels
    print(f"  new L2 distribution: {dict(pd.Series(labels).value_counts().sort_index())}", flush=True)

    # centroid を **元の標準化空間** で保存 (解釈性のため)
    K = int(labels.max() + 1)
    centroids_full = np.zeros((K, X_full.shape[1]))
    for c in range(K):
        centroids_full[c] = X_full[labels == c].mean(axis=0)
    # 圧縮空間 centroid も別途保持 (UMAP 用)
    centroids = np.zeros((K, X.shape[1]))
    for c in range(K):
        centroids[c] = X[labels == c].mean(axis=0)

    # 2D 座標 (PCA(n=2) は標準化済み全特徴に対して、UMAP は圧縮済み空間で)
    print("[2d] computing PCA/UMAP positions", flush=True)
    positions = _compute_2d_positions(centroids_full, X_full, labels)
    pca_explained = positions["pca"]["explained_variance_ratio"]
    print(f"  PCA(2D) explained variance: PC1={pca_explained[0]:.3f}, PC2={pca_explained[1]:.3f}", flush=True)

    # super-group (PCA(n=8) 圧縮後 centroid でコサイン距離 → 階層クラスタリング)
    print(f"[super] hierarchical grouping into {n_super} super-clusters", flush=True)
    sg_labels, sg_meta = _build_super_groups(centroids, n_super=n_super)
    print(f"  super-group sizes: {dict(pd.Series(sg_labels).value_counts().sort_index())}", flush=True)

    # 類似度行列 (PCA(n=8) 圧縮後 centroid ベース → 安定)
    sim = _compute_similarity(centroids)

    # ── 保存 ──
    # 1) unified.parquet に L2 を書き戻し
    uni.loc[sub.index, "L2"] = sub["L2_new"].astype(int)
    uni.loc[~mask_main, "L2"] = -1
    (ART_DIR / "unified.parquet").unlink(missing_ok=True)
    uni.to_parquet(ART_DIR / "unified.parquet")

    # 2) profiles (各クラスタの平均 lift)
    rel_with_l2 = rel.copy()
    rel_with_l2["L2"] = labels
    profiles = {}
    for L2 in sorted(set(labels)):
        means = rel_with_l2[rel_with_l2["L2"] == L2][COND_COLS_BASE].mean()
        profiles[int(L2)] = {c: round(float(means[c] - 1.0), 4) for c in COND_COLS_BASE}
    (ART_DIR / "l2_profiles.json").write_text(
        json.dumps({str(k): v for k, v in profiles.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) centroids (標準化済み生ベクトル 31D + PCA 圧縮後 8D)
    (ART_DIR / "l2_centroids.json").write_text(
        json.dumps({
            "features": COND_COLS_BASE,
            "centroids_full": {str(c): centroids_full[c].round(4).tolist() for c in range(K)},
            "centroids_reduced": {str(c): centroids[c].round(4).tolist() for c in range(K)},
            "n_features_full": int(X_full.shape[1]),
            "n_features_reduced": int(X.shape[1]),
            "pca_cumulative_variance": cum_var,
            "K": int(K),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) 2D 座標 (entity_id → 座標)
    stallion_pca = positions["pca"]["stallions"]
    pos_obj = {
        "pca": {
            "explained_variance_ratio": pca_explained,
            "centroids": {str(c): positions["pca"]["centroids"][c] for c in range(K)},
            "stallions": {},
        },
        "umap": {
            "centroids": {},
            "stallions": {},
        },
    }
    for i, (idx, row) in enumerate(sub.iterrows()):
        pos_obj["pca"]["stallions"][str(row["entity_id"])] = stallion_pca[i]
    if "centroids" in positions["umap"]:
        for c in range(K):
            pos_obj["umap"]["centroids"][str(c)] = positions["umap"]["centroids"][c]
        for i, (idx, row) in enumerate(sub.iterrows()):
            pos_obj["umap"]["stallions"][str(row["entity_id"])] = positions["umap"]["stallions"][i]
        pos_obj["umap"]["params"] = positions["umap"]["params"]
    else:
        pos_obj["umap"]["error"] = positions["umap"].get("error", "?")
    (ART_DIR / "l2_positions_2d.json").write_text(
        json.dumps(pos_obj, ensure_ascii=False), encoding="utf-8",
    )

    # 5) super-group マッピング + 構成
    sg_map: dict[str, dict[str, Any]] = {}
    for sg in sorted(set(sg_labels)):
        member_L2s = [int(c) for c in range(K) if sg_labels[c] == sg]
        # 各 super-group の centroid (member L2 centroid の平均)
        sg_centroid = centroids[[c for c in member_L2s]].mean(axis=0)
        sg_map[str(int(sg))] = {
            "member_L2s": member_L2s,
            "n_L2": len(member_L2s),
            "n_stallions": int(sum((labels == c).sum() for c in member_L2s)),
            "centroid": sg_centroid.round(4).tolist(),
        }
    (ART_DIR / "l2_super_groups.json").write_text(
        json.dumps({
            "super_groups": sg_map,
            "L2_to_super": {str(c): int(sg_labels[c]) for c in range(K)},
            "meta": sg_meta,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 6) similarity
    (ART_DIR / "l2_similarity.json").write_text(
        json.dumps(sim, ensure_ascii=False), encoding="utf-8",
    )

    # 7) scaler 情報
    (ART_DIR / "scaler_strength.json").write_text(json.dumps({
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": COND_COLS_BASE,
        "n_clusters": int(K),
        "n_super_groups": int(n_super),
        "silhouette_scores": sil_scores,
        "method": f"{method}_lift_normalized",
        "k_selection_range": [k_min, k_max],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[save] artifacts written to {ART_DIR}", flush=True)
    return {
        "K_fine": int(K),
        "n_super": int(n_super),
        "n_main": int(mask_main.sum()),
        "silhouette_scores": sil_scores,
        "pca_explained": pca_explained,
        "fine_distribution": {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "super_distribution": {int(k): int(v) for k, v in pd.Series(sg_labels).value_counts().sort_index().items()},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k_min", type=int, default=10)
    p.add_argument("--k_max", type=int, default=18)
    p.add_argument("--n_super", type=int, default=5)
    p.add_argument("--method", type=str, default="agglomerative",
                   choices=["agglomerative", "kmeans"])
    p.add_argument("--k", type=int, default=None, help="クラスタ数を固定")
    args = p.parse_args()
    res = build(k_min=args.k_min, k_max=args.k_max, n_super=args.n_super,
                method=args.method, k_fix=args.k)
    # 配列はサマリ出力で省略
    print(json.dumps({k: v for k, v in res.items() if k not in ("silhouette_scores",)},
                     ensure_ascii=False, indent=2))
