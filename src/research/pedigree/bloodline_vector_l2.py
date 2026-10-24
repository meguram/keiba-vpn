"""
血統ベクトル空間 v2 (L2 グループ単位)

これまでの血統分析資産を最大限活用したバージョン:
  - L2 メタクラスタ (主流血統からの派生 11 サブグループ)
  - 非主流血統 (root 単位の 6 グループ)
  - 31 次元の条件別 EB 勝率プロファイル (会場 × 馬場 × 距離 × ペース × コース形態)

旧版 (`bloodline_vector.py`) との違い:
  * ノードを「個別種牡馬」から「L2 サブグループ + 非主流ルート」(17 ノード) へ集約
  * 配置基底に「経験ベイズ収縮済みの条件別勝率」を採用 (使われ方バイアスを排除)
  * 類似度モードを 3 種類 (適性 / 構造 / ブレンド) で提供
  * 大系統 / Super Group / 主流非主流 など複数の色モード対応

Usage:
  python -m src.research.pedigree.bloodline_vector_l2
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger("research.pedigree.bloodline_vector_l2")

META_DIR = Path("data/page_reference/note_aptitude_race")
OUT_DIR = Path("data/page_reference/bloodline_vector/v2_l2")


# ─────────────────────────────────────────────────────
# 構造類似度の重み (main_group 一致 / super_group 一致)
# ─────────────────────────────────────────────────────

_STRUCT_SAME_SUPER_W = 1.0
_STRUCT_SAME_MAIN_W = 0.55
_STRUCT_DIFF_W = 0.0


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_l2_names(l2_names_blob: dict) -> dict[int, dict]:
    """l2_names.json -> {L2_id(int): meta_dict}"""
    raw = l2_names_blob.get("l2_names", {}) or {}
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def _flatten_super_groups(sg_blob: dict) -> tuple[dict[int, dict], dict[int, int]]:
    raw = sg_blob.get("super_groups", {}) or {}
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    l2_to_super_raw = sg_blob.get("L2_to_super", {}) or {}
    l2_to_super: dict[int, int] = {}
    for k, v in l2_to_super_raw.items():
        try:
            l2_to_super[int(k)] = int(v)
        except Exception:
            continue
    return out, l2_to_super


def _safe_float(x) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0


class BloodlineVectorL2Builder:
    """L2 サブグループ + 非主流グループのベクトル空間を構築する。"""

    def __init__(self, meta_dir: Path = META_DIR, output_dir: Path = OUT_DIR):
        self.meta_dir = Path(meta_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────
    def load_artifacts(self) -> None:
        self.unified = pd.read_parquet(self.meta_dir / "unified.parquet")
        self.l2_names = _flatten_l2_names(_safe_load_json(self.meta_dir / "l2_names.json"))
        super_blob = _safe_load_json(self.meta_dir / "l2_super_groups.json")
        self.super_groups, self.l2_to_super = _flatten_super_groups(super_blob)
        self.non_main_labels: dict[str, str] = _safe_load_json(
            self.meta_dir / "non_main_group_labels.json")
        self.sid_to_main: dict[str, str] = _safe_load_json(
            self.meta_dir / "sid_to_main.json")

        # 条件カラム (win_X) を抽出。win_eb_total はメタなので除外
        self.cond_cols = [
            c for c in self.unified.columns
            if c.startswith("win_") and c != "win_eb_total"
        ]
        logger.info("条件カラム: %d 個", len(self.cond_cols))
        logger.info("unified.parquet 行数: %d (stallion=%d, group=%d)",
                    len(self.unified),
                    int((self.unified.entity_type == "stallion").sum()),
                    int((self.unified.entity_type == "group").sum()))

    # ─────────────────────────────────────────────────
    def build_node_matrix(self) -> tuple[pd.DataFrame, np.ndarray]:
        """17 ノード × 31 次元の z 化済み行列と node_meta DataFrame を返す。"""
        df = self.unified.copy()
        # NaN を列平均で埋める
        for c in self.cond_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        col_mean = df[self.cond_cols].mean(axis=0, skipna=True)
        df[self.cond_cols] = df[self.cond_cols].fillna(col_mean)

        # 全エンティティで z 化 (stallion と group を共通スケールに揃える)
        scaler = StandardScaler()
        Z = scaler.fit_transform(df[self.cond_cols].values)
        df_z = pd.DataFrame(Z, index=df.index, columns=self.cond_cols)
        df_z["entity_id"] = df["entity_id"].values
        df_z["entity_type"] = df["entity_type"].values
        df_z["entity_label"] = df["entity_label"].values
        df_z["L2"] = df["L2"].values
        df_z["main_group"] = df["main_group"].values
        df_z["n_horses"] = df["n_horses"].values

        # ── L2 centroid (11) = stallion 行を L2 ID で平均
        st = df_z[df_z["entity_type"] == "stallion"]
        l2_ids = sorted([k for k in self.l2_names.keys() if k >= 0])
        l2_rows: list[dict] = []
        for lid in l2_ids:
            sub = st[st["L2"] == lid]
            if sub.empty:
                continue
            vec = sub[self.cond_cols].mean(axis=0).values
            meta = self.l2_names.get(lid, {})
            super_id = self.l2_to_super.get(lid, -1)
            super_name = (self.super_groups.get(super_id) or {}).get(
                "name", f"Super{super_id}")
            n_stallions = int((st["L2"] == lid).sum())
            l2_rows.append({
                "node_id": f"L2_{lid}",
                "node_type": "l2",
                "L2_id": lid,
                "super_id": super_id,
                "super_name": super_name,
                "name": meta.get("name", f"L2 {lid}"),
                "subtitle": meta.get("subtitle", ""),
                "color_hint": meta.get("color", "#888"),
                "icon": meta.get("icon", "●"),
                "n_members": int(meta.get("n_members", n_stallions)),
                "top_members": list(meta.get("top_members") or []),
                "highlights": list(meta.get("highlights") or []),
                "weaknesses": list(meta.get("weaknesses") or []),
                "rationale": meta.get("rationale", ""),
                "main_group": "派生 (主流)",
                "is_non_main": False,
                "vec": vec,
            })

        # ── 非主流グループ (6) = group 行をそのまま使用
        gr = df_z[df_z["entity_type"] == "group"]
        nm_rows: list[dict] = []
        for _, r in gr.iterrows():
            entity_id = r["entity_id"]
            entity_label = r["entity_label"] or self.non_main_labels.get(
                entity_id, entity_id)
            vec = r[self.cond_cols].values.astype(float)
            # 非主流の super_group は -1 (独立扱い)
            nm_rows.append({
                "node_id": f"NM_{entity_id}",
                "node_type": "non_main",
                "L2_id": -1,
                "super_id": -1,
                "super_name": "非主流",
                "name": entity_label,
                "subtitle": "非主流ルート派生グループ",
                "color_hint": "#94a3b8",
                "icon": "◇",
                "n_members": int(r["n_horses"]) if pd.notna(r["n_horses"]) else 0,
                "top_members": [],
                "highlights": [],
                "weaknesses": [],
                "rationale": "",
                "main_group": "非主流",
                "is_non_main": True,
                "vec": vec,
            })

        all_rows = l2_rows + nm_rows
        node_meta = pd.DataFrame([{k: v for k, v in r.items() if k != "vec"} for r in all_rows])
        M = np.vstack([r["vec"] for r in all_rows])
        logger.info("ノード行列 構築: %d nodes × %d dims (L2=%d, 非主流=%d)",
                    M.shape[0], M.shape[1], len(l2_rows), len(nm_rows))
        return node_meta, M

    # ─────────────────────────────────────────────────
    def compute_layouts(self, M: np.ndarray) -> dict[str, list[list[float]]]:
        layouts: dict[str, list[list[float]]] = {}

        # PCA (可逆 + 軸解釈用 loading)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        emb_pca = pca.fit_transform(M)
        layouts["pca"] = emb_pca.round(4).tolist()
        self._pca_loadings = {
            "components": pca.components_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "features": list(self.cond_cols),
        }

        # UMAP (cosine, n_neighbors=5 / min_dist=0.3 — 17 ノード前提)
        try:
            import umap  # type: ignore
            n = M.shape[0]
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(5, n - 1),
                min_dist=0.3,
                metric="cosine",
                random_state=42,
            )
            emb_umap = reducer.fit_transform(M)
            layouts["umap"] = emb_umap.round(4).tolist()
        except Exception as e:
            logger.warning("UMAP スキップ (%s) — PCA のみ", e)

        # t-SNE (任意)
        try:
            from sklearn.manifold import TSNE
            n = M.shape[0]
            tsne = TSNE(
                n_components=2,
                perplexity=min(5, max(2, n - 1)),
                metric="cosine",
                random_state=42,
                init="pca",
            )
            emb_tsne = tsne.fit_transform(M)
            layouts["tsne"] = emb_tsne.round(4).tolist()
        except Exception as e:
            logger.warning("t-SNE スキップ (%s)", e)

        return layouts

    # ─────────────────────────────────────────────────
    def compute_similarities(self, node_meta: pd.DataFrame, M: np.ndarray
                             ) -> dict[str, list[list[float]]]:
        from sklearn.metrics.pairwise import cosine_similarity
        # 適性類似 (31次元プロファイルの cos)
        sim_apt = cosine_similarity(M)

        # 構造類似 (super_group / main_group 一致)
        n = len(node_meta)
        sim_struct = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_struct[i, j] = 1.0
                    continue
                a = node_meta.iloc[i]
                b = node_meta.iloc[j]
                if (not a["is_non_main"] and not b["is_non_main"]
                        and a["super_id"] == b["super_id"]):
                    sim_struct[i, j] = _STRUCT_SAME_SUPER_W
                elif a["main_group"] == b["main_group"]:
                    sim_struct[i, j] = _STRUCT_SAME_MAIN_W
                else:
                    sim_struct[i, j] = _STRUCT_DIFF_W

        # ブレンド (0.6 適性 + 0.4 構造)
        sim_blend = 0.6 * sim_apt + 0.4 * sim_struct

        return {
            "aptitude": sim_apt.round(4).tolist(),
            "structure": sim_struct.round(4).tolist(),
            "blend": sim_blend.round(4).tolist(),
        }

    # ─────────────────────────────────────────────────
    def derive_aptitude_axes(self, node_meta: pd.DataFrame, M: np.ndarray) -> pd.DataFrame:
        """各ノードに対し、距離/馬場/道悪/コース形態の代表値を付与して
        UI の色塗りに使えるカラムを返す。"""
        cols = self.cond_cols
        col_idx = {c: i for i, c in enumerate(cols)}

        def safe_get(row_vec: np.ndarray, name: str) -> float:
            i = col_idx.get(name)
            return float(row_vec[i]) if i is not None else 0.0

        records: list[dict] = []
        for i in range(M.shape[0]):
            v = M[i]
            sprint = safe_get(v, "win_d_短距離")
            mile = safe_get(v, "win_d_マイル")
            mid = safe_get(v, "win_d_中距離")
            stay = safe_get(v, "win_d_長距離")
            turf = safe_get(v, "win_s_芝")
            dirt = safe_get(v, "win_s_ダート")
            heavy = safe_get(v, "win_heavy")
            steep = safe_get(v, "win_steep")
            flat = safe_get(v, "win_flat")
            # 距離支配 (4 つで argmax)
            dist_vals = [sprint, mile, mid, stay]
            dist_label = ["短距離", "マイル", "中距離", "長距離"][int(np.argmax(dist_vals))]
            # 馬場
            surf_label = "芝" if turf >= dirt else "ダート"
            # 道悪
            heavy_label = "道悪巧者" if heavy > 0.3 else (
                "道悪苦手" if heavy < -0.3 else "中立")
            records.append({
                "z_sprint": round(sprint, 3),
                "z_mile": round(mile, 3),
                "z_mid": round(mid, 3),
                "z_stay": round(stay, 3),
                "z_turf": round(turf, 3),
                "z_dirt": round(dirt, 3),
                "z_heavy": round(heavy, 3),
                "z_steep": round(steep, 3),
                "z_flat": round(flat, 3),
                "dist_dominant": dist_label,
                "surface_dominant": surf_label,
                "heavy_label": heavy_label,
            })
        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────
    def save(self, node_meta: pd.DataFrame, layouts: dict, sims: dict,
             aptitude_axes: pd.DataFrame) -> None:
        # ノード一覧 + 配置 + 適性軸サマリ
        nodes: list[dict] = []
        for i, row in node_meta.iterrows():
            ax = aptitude_axes.iloc[i]
            nodes.append({
                "idx": int(i),
                "node_id": row["node_id"],
                "node_type": row["node_type"],
                "name": row["name"],
                "subtitle": row["subtitle"],
                "L2_id": int(row["L2_id"]) if row["L2_id"] != -1 else None,
                "super_id": int(row["super_id"]) if row["super_id"] != -1 else None,
                "super_name": row["super_name"],
                "main_group": row["main_group"],
                "is_non_main": bool(row["is_non_main"]),
                "color_hint": row["color_hint"],
                "icon": row["icon"],
                "n_members": int(row["n_members"]),
                "top_members": list(row["top_members"]),
                "highlights": list(row["highlights"]),
                "weaknesses": list(row["weaknesses"]),
                "rationale": row["rationale"],
                "pos": {k: layouts.get(k, [[0, 0]] * len(node_meta))[i] for k in layouts},
                "axes": {
                    "dist_dominant": ax["dist_dominant"],
                    "surface_dominant": ax["surface_dominant"],
                    "heavy_label": ax["heavy_label"],
                    "z_sprint": ax["z_sprint"],
                    "z_mile": ax["z_mile"],
                    "z_mid": ax["z_mid"],
                    "z_stay": ax["z_stay"],
                    "z_turf": ax["z_turf"],
                    "z_dirt": ax["z_dirt"],
                    "z_heavy": ax["z_heavy"],
                    "z_steep": ax["z_steep"],
                    "z_flat": ax["z_flat"],
                },
            })

        # 各ノードに対する類似上位 8 件を type 別に算出
        n = len(nodes)
        sim_top: dict[str, dict[str, list[dict]]] = {"aptitude": {}, "structure": {}, "blend": {}}
        for mode_name, mat in sims.items():
            arr = np.array(mat)
            for i in range(n):
                row = arr[i]
                order = np.argsort(-row)
                top_list = []
                for j in order:
                    if j == i:
                        continue
                    top_list.append({
                        "node_id": nodes[j]["node_id"],
                        "name": nodes[j]["name"],
                        "score": float(round(row[j], 4)),
                    })
                    if len(top_list) >= 8:
                        break
                sim_top[mode_name][nodes[i]["node_id"]] = top_list

        # PCA loadings (軸ラベル用)
        loadings_summary: dict[str, list[dict]] = {}
        if hasattr(self, "_pca_loadings"):
            comps = self._pca_loadings["components"]
            feats = self._pca_loadings["features"]
            for axis_idx, comp in enumerate(comps):
                pairs = sorted(zip(feats, comp), key=lambda p: -abs(p[1]))[:5]
                loadings_summary[f"pc{axis_idx + 1}"] = [
                    {"feature": f, "weight": round(w, 3)} for f, w in pairs
                ]

        meta_out = {
            "version": "v2_l2_2026_05",
            "n_nodes": n,
            "feature_dim": len(self.cond_cols),
            "features": list(self.cond_cols),
            "layouts_available": list(layouts.keys()),
            "pca_loadings_top5": loadings_summary,
            "explained_variance_ratio": (
                self._pca_loadings.get("explained_variance_ratio")
                if hasattr(self, "_pca_loadings") else None),
            "super_groups": {
                str(k): {
                    "name": v.get("name"),
                    "color": v.get("color"),
                    "icon": v.get("icon"),
                } for k, v in self.super_groups.items()
            },
            "main_group_palette": {
                "派生 (主流)": "#60a5fa",
                "非主流": "#94a3b8",
            },
        }

        out_nodes_path = self.output_dir / "groups_embeddings.json"
        out_nodes_path.write_text(
            json.dumps({"nodes": nodes, "similar_top": sim_top, "meta": meta_out},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        logger.info("保存: %s", out_nodes_path)

        # 類似度行列 (デバッグ・将来用)
        sim_path = self.output_dir / "groups_similarity.json"
        sim_path.write_text(json.dumps(sims, ensure_ascii=False), encoding="utf-8")
        logger.info("保存: %s", sim_path)

    # ─────────────────────────────────────────────────
    def run(self) -> dict:
        self.load_artifacts()
        node_meta, M = self.build_node_matrix()
        layouts = self.compute_layouts(M)
        sims = self.compute_similarities(node_meta, M)
        axes = self.derive_aptitude_axes(node_meta, M)
        self.save(node_meta, layouts, sims, axes)
        return {
            "n_nodes": int(M.shape[0]),
            "feature_dim": int(M.shape[1]),
            "layouts": list(layouts.keys()),
        }


def main() -> None:
    script_basic_config()
    builder = BloodlineVectorL2Builder()
    summary = builder.run()
    print("=== bloodline_vector_l2 build summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
