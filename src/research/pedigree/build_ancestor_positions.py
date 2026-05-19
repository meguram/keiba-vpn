"""全祖先 (4,773) の 2D 位置 (PCA / UMAP) と L2 再割当を計算する。

前段の ``build_ancestor_vectors`` で生成した ``ancestor_vectors.parquet`` を
入力とし、PCA / UMAP で 2D 位置を計算する。また L2 を centroid (cosine) 最近接で
再判定し、より精密な L2 ラベルに更新する。

入出力:
    in:  data/research/bloodline_meta_cluster/ancestor_vectors.parquet
         data/research/bloodline_meta_cluster/l2_centroids.json
         data/research/bloodline_meta_cluster/scaler_strength.json
    out: data/research/bloodline_meta_cluster/ancestor_positions_2d.json
            {"pca": {sid: [x, y]}, "umap": {sid: [x, y]}}
         data/research/bloodline_meta_cluster/ancestor_to_l2.json (上書き)
            method = "vector_centroid" を新規追加

Usage:
    python -m src.research.pedigree.build_ancestor_positions
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/research/bloodline_meta_cluster"

FEAT_COLS = [
    "win_v_東京", "win_v_中山", "win_v_阪神", "win_v_京都", "win_v_中京",
    "win_v_新潟", "win_v_小倉", "win_v_福島", "win_v_札幌", "win_v_函館",
    "win_s_芝", "win_s_ダート",
    "win_d_短距離", "win_d_マイル", "win_d_中距離", "win_d_長距離",
    "win_pace_スロー瞬発力_短距離", "win_pace_スロー瞬発力_マイル",
    "win_pace_スロー瞬発力_中距離", "win_pace_スロー瞬発力_長距離",
    "win_pace_持続力勝負_短距離", "win_pace_持続力勝負_マイル",
    "win_pace_持続力勝負_中距離", "win_pace_持続力勝負_長距離",
    "win_pace_持久力勝負_短距離", "win_pace_持久力勝負_マイル",
    "win_pace_持久力勝負_中距離", "win_pace_持久力勝負_長距離",
    "win_steep", "win_flat", "win_heavy",
]


def main() -> int:
    print("[load] ancestor_vectors.parquet ...", flush=True)
    df = pd.read_parquet(ART / "ancestor_vectors.parquet")
    df["stallion_id"] = df["stallion_id"].astype(str)
    X = df[FEAT_COLS].values.astype(np.float64)
    print(f"  X shape: {X.shape}")

    # ── 0) pace カラム (12 列) は ancestor_vectors では集計されていない (全 0) →
    # cosine 比較で誤った 0 への引き込みを避けるため、L2 再判定では除外する
    pace_mask = np.array([not c.startswith("win_pace_") for c in FEAT_COLS])

    # ── 1) PCA (2D) ──
    print("[pca] 2D ...", flush=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(Xs)
    print(f"  explained_variance: {pca.explained_variance_ratio_.tolist()}")

    # ── 2) UMAP (2D) ──
    try:
        import umap
        print("[umap] 2D (4,773 頭は少し時間がかかります) ...", flush=True)
        # 8D に PCA 縮約してから UMAP (高速化 & 安定化)
        pca8 = PCA(n_components=8, random_state=42)
        X8 = pca8.fit_transform(Xs)
        reducer = umap.UMAP(
            n_neighbors=20, min_dist=0.1, random_state=42, metric="cosine",
        )
        X_umap = reducer.fit_transform(X8)
        umap_params = {"n_neighbors": 20, "min_dist": 0.1, "metric": "cosine"}
    except ImportError:
        print("  [warn] umap-learn 未インストール → UMAP スキップ")
        X_umap = None
        umap_params = None

    # ── 3) L2 を centroid 最近接で再判定 ──
    # 注意: L2 centroid は scaler_strength.json で z-score 化された値。
    #       これを raw 勝率空間に逆変換してから ancestor_vectors と同じ空間で
    #       cosine を取る。さらに pace 12 列 (ancestor_vectors では全 0) は除外。
    print("[L2 reassign] pace 除外 + raw 空間で cosine 最近接 ...", flush=True)
    centroids = json.load(open(ART / "l2_centroids.json", encoding="utf-8"))
    cent_full_dict = centroids["centroids_full"]   # {"0": [31D z-score], ...}
    L2_ids = sorted([int(k) for k in cent_full_dict])
    C_z = np.array([cent_full_dict[str(i)] for i in L2_ids], dtype=np.float64)
    scaler_strength = json.load(open(ART / "scaler_strength.json", encoding="utf-8"))
    s_mean = np.array(scaler_strength["mean"], dtype=np.float64)
    s_scale = np.array(scaler_strength["scale"], dtype=np.float64)
    C_raw = C_z * s_scale + s_mean      # raw 勝率空間に戻した centroid

    # pace 除外で再度 z-score 化 (mask 後の特徴で標準化し直し)
    X_m = X[:, pace_mask]
    C_m = C_raw[:, pace_mask]
    mean_m = s_mean[pace_mask]
    scale_m = s_scale[pace_mask]
    Xs_m = (X_m - mean_m) / scale_m
    Cs_m = (C_m - mean_m) / scale_m

    def _cos(a, B):
        an = np.linalg.norm(a) + 1e-12
        bn = np.linalg.norm(B, axis=1) + 1e-12
        return (B @ a) / (an * bn)

    new_L2 = []
    new_score = []
    for i in range(Xs_m.shape[0]):
        sims = _cos(Xs_m[i], Cs_m)
        idx = int(np.argmax(sims))
        new_L2.append(L2_ids[idx])
        new_score.append(float(sims[idx]))

    df["L2_new"] = new_L2
    df["L2_score"] = new_score

    # ── 4) ancestor_to_l2.json には L2_score (信頼度) を追記するのみ。
    #       L2 ラベル自体は子孫票決ベース (build_ancestor_l2_index の出力) を維持する。
    #       理由: pace 12 列が ancestor_vectors では全 0 のため、cosine 再判定は
    #             pace 軸の損失で L2=4 (東京型) 等に偏る現象が観測される。
    anc_l2_path = ART / "ancestor_to_l2.json"
    anc_l2 = json.load(open(anc_l2_path, encoding="utf-8"))
    for _, r in df.iterrows():
        sid = r["stallion_id"]
        if sid not in anc_l2:
            anc_l2[sid] = {"L2": int(r["L2_new"]), "method": "vector_only"}
            continue
        anc_l2[sid]["L2_nearest"] = int(r["L2_new"])       # 参考: 最近接 centroid
        anc_l2[sid]["L2_score"] = float(r["L2_score"])     # 参考: 類似度
    json.dump(anc_l2, open(anc_l2_path, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"  ancestor_to_l2.json updated (L2_nearest + L2_score appended)")

    # ── 5) 2D 位置 JSON 出力 ──
    out_2d = {
        "pca": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "positions": {
                str(r["stallion_id"]): [float(X_pca[i, 0]), float(X_pca[i, 1])]
                for i, (_, r) in enumerate(df.iterrows())
            },
        },
    }
    if X_umap is not None:
        out_2d["umap"] = {
            "params": umap_params,
            "positions": {
                str(r["stallion_id"]): [float(X_umap[i, 0]), float(X_umap[i, 1])]
                for i, (_, r) in enumerate(df.iterrows())
            },
        }
    out_path = ART / "ancestor_positions_2d.json"
    json.dump(out_2d, open(out_path, "w", encoding="utf-8"))
    print(f"[save] {out_path}  ({len(df)} stallions)", flush=True)

    # ── 6) ancestor_vectors.parquet に L2_nearest と L2_score を追記 ──
    df["L2_nearest"] = df["L2_new"]
    df = df.drop(columns=["L2_new"])
    # L2 列 (子孫票決ベース) も併記 (ancestor_to_l2 から)
    df["L2"] = df["stallion_id"].map(lambda s: anc_l2.get(s, {}).get("L2", -1))
    df.to_parquet(ART / "ancestor_vectors.parquet")
    print(f"[update] ancestor_vectors.parquet (L2 + L2_nearest + L2_score)")

    print("\n=== L2 分布 (子孫票決ベース = 採用) ===")
    print(df["L2"].value_counts().sort_index().to_dict())
    print("=== L2_nearest 分布 (cosine 最近接 = 参考) ===")
    print(df["L2_nearest"].value_counts().sort_index().to_dict())
    print(f"\n=== L2_score 統計 ===")
    print(f"  mean: {df['L2_score'].mean():.3f}")
    print(f"  median: {df['L2_score'].median():.3f}")
    print(f"  > 0.5 (信頼度高): {(df['L2_score'] > 0.5).sum()} 頭")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
