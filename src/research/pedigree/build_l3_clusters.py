"""L3 (細粒度適性) クラスタリング: 3 視点 × 条件別 KPI + タグ Boolean のハイブリッド特徴量。

L2 (大カテゴリ 4 個) の中で、各種牡馬を「父視点 / 母父視点 / 母母視点」の 3 視点別に評価し、
さらに細かい「得意分野クラスタ L3」を抽出する。

特徴量設計 (1 種牡馬 × 1 視点 あたり):
    [A] 条件別 lift (31 次元): 各種牡馬のその視点での "条件別勝率 / 視点全体勝率"
    [B] タグ Boolean (28 次元): その種牡馬 sire 視点で is_top20 のタグを 1/0

クラスタリング:
    - 視点ごとに独立 (father / dam_sire_line / dam_dam_line で別)
    - 各 L2 内で再分割 (h_nested 階層構造)
    - Ward 法 + シルエットスコアで最適 k を選定 (k ∈ [3, 8])

出力:
    data/research/bloodline_meta_cluster_l3/
        father_l3.parquet         : stallion_id, L2, L3, n_records_father
        dam_sire_line_l3.parquet  : 同上 (dam_sire_line 視点)
        dam_dam_line_l3.parquet   : 同上
        l3_profiles.json          : 各 (view, L2, L3) の特徴量平均値
        l3_meta.json              : 構築メタ情報

Usage:
    python -m src.research.pedigree.build_l3_clusters [--scope main|all]
        main: 現 L2 メンバ約 130 種牡馬 (プロトタイプ)
        all : ancestor_to_horses から子孫 >= 100 頭の祖先全数 (Phase 2 用)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

ART_DIR = ROOT / "data/research/bloodline_meta_cluster"
IDX_3V = ROOT / "data/research/pedigree_10gen_3view"
OUT_DIR = ROOT / "data/research/bloodline_meta_cluster_l3"

# 31 条件カラム (建立元の特徴量と同じ)
COND_COLS = [
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


def _classify_record(rec: pd.DataFrame) -> pd.DataFrame:
    """race_records に dist_cat 等の派生カラムを付与 (簡易版)。"""
    df = rec.copy()
    if "dist_cat" not in df.columns and "distance" in df.columns:
        df["dist_cat"] = pd.cut(
            df["distance"], bins=[0, 1400, 1800, 2200, 9999],
            labels=["短距離", "マイル", "中距離", "長距離"],
        ).astype(str)
    for c in ("venue", "surface", "track_condition", "dist_cat"):
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")
    return df


STEEP_VENUES_LIST = ["中山", "阪神", "中京"]
FLAT_VENUES_LIST = ["東京", "新潟", "京都"]
HEAVY_TC_LIST = ["重", "不良", "稍重"]


def _build_race_index(race: pd.DataFrame) -> dict[str, Any]:
    """race_records を numpy 配列に展開して高速 lookup できる構造を作る。"""
    horse_id_arr = race["horse_id"].astype(str).values
    horse_to_rows: dict[str, list[int]] = {}
    for i, h in enumerate(horse_id_arr):
        horse_to_rows.setdefault(h, []).append(i)
    horse_to_rows_np = {h: np.array(v, dtype=np.int32) for h, v in horse_to_rows.items()}
    return {
        "horse_to_rows": horse_to_rows_np,
        "finish": race["finish_position"].fillna(-1).astype(float).values,
        "venue": race["venue"].astype(str).values if "venue" in race.columns else np.array([""] * len(race)),
        "surface": race["surface"].astype(str).values if "surface" in race.columns else np.array([""] * len(race)),
        "dist_cat": race["dist_cat"].astype(str).values if "dist_cat" in race.columns else np.array([""] * len(race)),
        "track_condition": race["track_condition"].astype(str).values if "track_condition" in race.columns else np.array([""] * len(race)),
    }


def _compute_view_kpi(
    horse_ids: set[str],
    race_or_index,
) -> dict[str, Any]:
    """precompute index を使った高速 KPI 算出。

    race_or_index: dict (precompute) または DataFrame (fallback)
    """
    if isinstance(race_or_index, dict):
        rix = race_or_index
        horse_to_rows = rix["horse_to_rows"]
        row_lists = [horse_to_rows[h] for h in horse_ids if h in horse_to_rows]
        if not row_lists:
            return {"n_total": 0, "n_horses": 0, "win_eb_total": np.nan}
        rows = np.concatenate(row_lists)
        finish = rix["finish"][rows]
        mask = finish > 0
        if not mask.any():
            return {"n_total": 0, "n_horses": 0, "win_eb_total": np.nan}
        rows_f = rows[mask]
        finish_f = finish[mask]
        venue_f = rix["venue"][rows_f]
        surface_f = rix["surface"][rows_f]
        dist_cat_f = rix["dist_cat"][rows_f]
        tc_f = rix["track_condition"][rows_f]
        is_win = finish_f == 1
        n_total = int(len(rows_f))
        n_win = int(is_win.sum())
        out: dict[str, Any] = {
            "n_total": n_total,
            "n_horses": int(len(set(horse_ids))),
            "win_eb_total": float(n_win / n_total) if n_total else np.nan,
        }
        for v in ["東京", "中山", "阪神", "京都", "中京", "新潟", "小倉", "福島", "札幌", "函館"]:
            m = venue_f == v
            out[f"win_v_{v}"] = float(is_win[m].sum() / m.sum()) if m.any() else np.nan
        for s in ["芝", "ダート"]:
            m = surface_f == s
            out[f"win_s_{s}"] = float(is_win[m].sum() / m.sum()) if m.any() else np.nan
        for d in ["短距離", "マイル", "中距離", "長距離"]:
            m = dist_cat_f == d
            out[f"win_d_{d}"] = float(is_win[m].sum() / m.sum()) if m.any() else np.nan
        for pace in ["スロー瞬発力", "持続力勝負", "持久力勝負"]:
            for d in ["短距離", "マイル", "中距離", "長距離"]:
                out[f"win_pace_{pace}_{d}"] = np.nan
        m_steep = np.isin(venue_f, STEEP_VENUES_LIST)
        out["win_steep"] = float(is_win[m_steep].sum() / m_steep.sum()) if m_steep.any() else np.nan
        m_flat = np.isin(venue_f, FLAT_VENUES_LIST)
        out["win_flat"] = float(is_win[m_flat].sum() / m_flat.sum()) if m_flat.any() else np.nan
        m_heavy = np.isin(tc_f, HEAVY_TC_LIST)
        out["win_heavy"] = float(is_win[m_heavy].sum() / m_heavy.sum()) if m_heavy.any() else np.nan
        return out
    # fallback (DataFrame)
    race = race_or_index
    sub = race[race["horse_id"].isin(horse_ids)]
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]
    if sub.empty:
        return {"n_total": 0, "n_horses": 0, "win_eb_total": np.nan}
    n_total = len(sub)
    n_win = int((sub["finish_position"] == 1).sum())
    out = {
        "n_total": int(n_total),
        "n_horses": int(sub["horse_id"].nunique()),
        "win_eb_total": float(n_win / n_total) if n_total else np.nan,
    }
    for v in ["東京", "中山", "阪神", "京都", "中京", "新潟", "小倉", "福島", "札幌", "函館"]:
        s2 = sub[sub.get("venue") == v]
        out[f"win_v_{v}"] = float((s2["finish_position"] == 1).sum() / len(s2)) if len(s2) else np.nan
    for s in ["芝", "ダート"]:
        s2 = sub[sub.get("surface") == s]
        out[f"win_s_{s}"] = float((s2["finish_position"] == 1).sum() / len(s2)) if len(s2) else np.nan
    for d in ["短距離", "マイル", "中距離", "長距離"]:
        s2 = sub[sub.get("dist_cat") == d]
        out[f"win_d_{d}"] = float((s2["finish_position"] == 1).sum() / len(s2)) if len(s2) else np.nan
    for pace in ["スロー瞬発力", "持続力勝負", "持久力勝負"]:
        for d in ["短距離", "マイル", "中距離", "長距離"]:
            out[f"win_pace_{pace}_{d}"] = np.nan
    out["win_steep"] = float(
        (sub[sub.get("venue").isin(STEEP_VENUES_LIST)]["finish_position"] == 1).sum()
        / max(1, len(sub[sub.get("venue").isin(STEEP_VENUES_LIST)]))
    ) if any(sub.get("venue").isin(STEEP_VENUES_LIST)) else np.nan
    out["win_flat"] = float(
        (sub[sub.get("venue").isin(FLAT_VENUES_LIST)]["finish_position"] == 1).sum()
        / max(1, len(sub[sub.get("venue").isin(FLAT_VENUES_LIST)]))
    ) if any(sub.get("venue").isin(FLAT_VENUES_LIST)) else np.nan
    out["win_heavy"] = float(
        (sub[sub.get("track_condition").isin(HEAVY_TC_LIST)]["finish_position"] == 1).sum()
        / max(1, len(sub[sub.get("track_condition").isin(HEAVY_TC_LIST)]))
    ) if any(sub.get("track_condition").isin(HEAVY_TC_LIST)) else np.nan
    return out


def _load_target_stallions(
    scope: str, max_total_horses: int | None = None
) -> list[str]:
    if scope == "main":
        uni = pd.read_parquet(ART_DIR / "unified.parquet")
        sub = uni[(uni["entity_type"] == "stallion") & (uni["L2"] >= 0)]
        return [str(s) for s in sub["entity_id"].tolist()]
    elif scope in ("all", "niche"):
        inv = pd.read_parquet(IDX_3V / "ancestor_to_horses.parquet")
        s = inv.groupby("ancestor_id")["n_horses"].sum()
        sel = s[s >= 100]
        if max_total_horses is not None:
            sel = sel[sel <= max_total_horses]
        return [str(a) for a in sel.index]
    else:
        raise ValueError(f"unknown scope: {scope}")


def _load_horses_for_view(stallion_id: str, view: str, inv: pd.DataFrame) -> set[str]:
    """指定 stallion_id × view の該当馬 set を返す。"""
    sub = inv[(inv["ancestor_id"] == stallion_id) & (inv["side"] == view)]
    if sub.empty:
        return set()
    horse_ids_lol = sub["horse_ids"].iloc[0]
    if horse_ids_lol is None:
        return set()
    return {str(h) for h in horse_ids_lol}


def build_view_kpi_table(
    scope: str, view: str, min_n: int = 30, max_total_horses: int | None = None
) -> pd.DataFrame:
    """指定 view (father/dam_sire_line/dam_dam_line) で全対象種牡馬の KPI を集計。"""
    print(f"[L3] view={view} 集計開始 (scope={scope}, min_n={min_n}, max_total={max_total_horses})", flush=True)
    sids = _load_target_stallions(scope, max_total_horses=max_total_horses)
    print(f"[L3]  対象種牡馬: {len(sids):,}", flush=True)

    race = pd.read_parquet(ART_DIR / "race_records.parquet")
    race = _classify_record(race)
    race["horse_id"] = race["horse_id"].astype(str)

    # race index precompute (一括)
    print(f"[L3]  race index precompute...", flush=True)
    t0 = time.time()
    rix = _build_race_index(race)
    print(f"[L3]  race index done ({time.time()-t0:.1f}s, {len(rix['horse_to_rows']):,} horses)", flush=True)

    # main scope の father 視点: race_records.stallion_id 経由
    if view == "father" and scope == "main":
        h2s = pd.read_parquet(ART_DIR / "horse_to_sire.parquet")
        h2s["horse_id"] = h2s["horse_id"].astype(str)
        h2s["stallion_id"] = h2s["stallion_id"].astype(str)
        sire_of = dict(zip(h2s["horse_id"], h2s["stallion_id"]))
        race["stallion_id"] = race["horse_id"].map(sire_of)
        # stallion_id -> set of horse_ids
        sid_to_horses: dict[str, set[str]] = {}
        for h, s in sire_of.items():
            sid_to_horses.setdefault(s, set()).add(h)
        rows = []
        t0 = time.time()
        for i, sid in enumerate(sids):
            horse_set = sid_to_horses.get(sid, set())
            if len(horse_set) < min_n:
                continue
            kpi = _compute_view_kpi(horse_set, rix)
            if kpi.get("n_total", 0) < min_n:
                continue
            kpi["stallion_id"] = sid
            rows.append(kpi)
            if (i + 1) % 50 == 0:
                print(f"[L3]   {i+1}/{len(sids)} elapsed={time.time()-t0:.1f}s", flush=True)
        df = pd.DataFrame(rows)
        print(f"[L3]  view={view} 完了: {len(df)} 種牡馬 ({time.time()-t0:.1f}s)", flush=True)
        return df

    # 3view index 経由 (all scope または mother 系視点)
    inv_path = IDX_3V / "ancestor_to_horses.parquet"
    if not inv_path.exists():
        raise FileNotFoundError(f"3view インデックスが無い: {inv_path}")
    print(f"[L3]  inv index precompute...", flush=True)
    t0 = time.time()
    inv_df = pd.read_parquet(inv_path)
    inv_df = inv_df[inv_df["side"] == view]
    # ancestor_id -> set of horse_ids
    anc_to_horses: dict[str, set[str]] = {}
    for _, r in inv_df.iterrows():
        hids = r["horse_ids"]
        if hids is None or (hasattr(hids, "__len__") and len(hids) == 0):
            continue
        anc_to_horses[str(r["ancestor_id"])] = {str(h) for h in hids}
    print(f"[L3]  inv index done ({time.time()-t0:.1f}s, {len(anc_to_horses):,} ancestors)", flush=True)

    rows = []
    t0 = time.time()
    for i, sid in enumerate(sids):
        horse_set = anc_to_horses.get(sid, set())
        if len(horse_set) < min_n:
            continue
        kpi = _compute_view_kpi(horse_set, rix)
        if kpi.get("n_total", 0) < min_n:
            continue
        kpi["stallion_id"] = sid
        rows.append(kpi)
        if (i + 1) % 500 == 0:
            print(f"[L3]   {i+1}/{len(sids)} elapsed={time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    print(f"[L3]  view={view} 完了: {len(df)} 種牡馬 ({time.time()-t0:.1f}s)", flush=True)
    return df


def _attach_tags(df: pd.DataFrame) -> pd.DataFrame:
    """各種牡馬に「タグ Boolean ベクトル」を付与 (sire 視点の is_top20 を採用)。"""
    tags_path = ART_DIR / "stallion_tags_full.parquet"
    if not tags_path.exists():
        return df
    tdf = pd.read_parquet(tags_path)
    tdf["stallion_id"] = tdf["stallion_id"].astype(str)
    tdf = tdf[tdf["is_top20"]]
    tag_ids = sorted(tdf["tag_id"].unique())
    pivot = tdf.pivot_table(
        index="stallion_id", columns="tag_id",
        values="is_top20", aggfunc="first", fill_value=False,
    ).astype(int)
    pivot.columns = [f"tag_{c}" for c in pivot.columns]
    df = df.merge(pivot, left_on="stallion_id", right_index=True, how="left")
    for c in pivot.columns:
        df[c] = df[c].fillna(0).astype(int)
    return df


def _relativize(df: pd.DataFrame) -> pd.DataFrame:
    """条件別勝率を lift (= 条件勝率 / win_eb_total) に変換。"""
    out = df.copy()
    base = out["win_eb_total"].clip(lower=0.001)
    for c in COND_COLS:
        if c in out.columns:
            out[c] = (out[c] / base).fillna(1.0).clip(lower=0.3, upper=3.0)
    return out


def _merge_small_clusters(labels: np.ndarray, Xs: np.ndarray, min_size: int = 5) -> np.ndarray:
    """小さなクラスタを最近接の大きなクラスタに併合し、ラベルを 0..K-1 に再付与。"""
    labels = labels.copy()
    while True:
        unique = np.unique(labels)
        sizes = {int(c): int((labels == c).sum()) for c in unique}
        small = [c for c, n in sizes.items() if n < min_size]
        if not small:
            break
        # 最も小さいクラスタを選ぶ
        target = min(small, key=lambda c: sizes[c])
        large = [c for c in unique if sizes.get(int(c), 0) >= min_size]
        if not large:
            break
        # target の centroid → large の centroid 最小距離
        target_centroid = Xs[labels == target].mean(axis=0)
        d_min, best = float("inf"), None
        for c in large:
            cc = Xs[labels == c].mean(axis=0)
            d = np.linalg.norm(target_centroid - cc)
            if d < d_min:
                d_min, best = d, c
        if best is None:
            break
        labels[labels == target] = best
    # 0..K-1 に再付与
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.array([remap[c] for c in labels], dtype=int)


def cluster_view_within_l2(
    kpi: pd.DataFrame,
    view: str,
    k_per_l2_range: tuple[int, int] = (3, 6),
    min_l2_members: int = 8,
    min_l3_size: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """各 L2 内で L3 クラスタを再分割。

    kpi: 'stallion_id' + 'L2' + 31 条件 lift + tag_* Boolean 列を含む DataFrame

    k 選択: シルエット最大 + 最小クラスタサイズ >= 3 を満たすもの。
    その後、min_l3_size 未満のクラスタを最近接に併合。
    """
    rows = []
    profiles: dict[str, Any] = {}
    feat_cols_cond = [c for c in COND_COLS if c in kpi.columns]
    feat_cols_tag = [c for c in kpi.columns if c.startswith("tag_")]
    feat_cols = feat_cols_cond + feat_cols_tag

    for L2 in sorted(kpi["L2"].dropna().unique()):
        sub = kpi[kpi["L2"] == L2].copy()
        if len(sub) < min_l2_members:
            sub["L3"] = 0
            rows.append(sub[["stallion_id", "L2", "L3"]])
            profiles[f"{view}/L2={int(L2)}/L3=0"] = {"n_members": int(len(sub))}
            continue
        X = sub[feat_cols].fillna(0).values
        if X.shape[1] == 0:
            sub["L3"] = 0
            rows.append(sub[["stallion_id", "L2", "L3"]])
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # 最適 k 選定 (シルエット + 最小クラスタサイズ制約)
        kmax = min(k_per_l2_range[1], max(2, len(sub) // max(1, min_l3_size)))
        candidates = []
        for k in range(k_per_l2_range[0], kmax + 1):
            try:
                km = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = km.fit_predict(Xs)
                if len(set(labels)) < 2:
                    continue
                sizes = [int((labels == c).sum()) for c in set(labels)]
                if min(sizes) < 3:
                    # min 3 を満たさないなら候補から除外
                    continue
                sc = silhouette_score(Xs, labels)
                candidates.append((sc, k, labels))
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            sil_best, k_best, labels_best = candidates[0]
        else:
            # フォールバック: k=2 で
            try:
                km = AgglomerativeClustering(n_clusters=2, linkage="ward")
                labels_best = km.fit_predict(Xs)
                sil_best = silhouette_score(Xs, labels_best) if len(set(labels_best)) > 1 else 0.0
                k_best = 2
            except Exception:
                labels_best = np.zeros(len(sub), dtype=int)
                sil_best = 0.0
                k_best = 1
        # 外れ値 merge
        labels_merged = _merge_small_clusters(labels_best, Xs, min_size=min_l3_size)
        sub["L3"] = labels_merged
        rows.append(sub[["stallion_id", "L2", "L3"]])
        for L3 in sorted(set(labels_merged)):
            mask = labels_merged == L3
            key = f"{view}/L2={int(L2)}/L3={int(L3)}"
            profile = {
                "n_members": int(mask.sum()),
                "silhouette": float(sil_best),
                "k": int(k_best),
                "k_final": int(len(set(labels_merged))),
                "cond_lift_mean": {c: round(float(sub.loc[mask, c].mean()), 3) for c in feat_cols_cond[:31]},
                "tag_freq": {c.replace("tag_", ""): round(float(sub.loc[mask, c].mean()), 3) for c in feat_cols_tag},
                "top_members": sub.loc[mask].sort_values("n_total", ascending=False).head(8)["stallion_id"].tolist(),
            }
            profiles[key] = profile
    return pd.concat(rows, axis=0).reset_index(drop=True), profiles


def _expand_l2_assignment(kpi: pd.DataFrame, sid_to_l2: dict[str, int]) -> pd.DataFrame:
    """既存 L2 (133 種牡馬) を学習データとして、未登録祖先を centroid 距離で最近接 L2 に割り当て。

    特徴量: 31 条件 lift + win_eb_total。
    """
    kpi = kpi.copy()
    kpi["L2"] = kpi["stallion_id"].map(sid_to_l2)
    feat_cols = [c for c in COND_COLS if c in kpi.columns]
    if not feat_cols:
        kpi["L2"] = kpi["L2"].fillna(-1).astype(int)
        return kpi

    train = kpi.dropna(subset=["L2"]).copy()
    if train.empty:
        kpi["L2"] = kpi["L2"].fillna(-1).astype(int)
        return kpi

    X_train = train[feat_cols].fillna(1.0).values
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    centroids: dict[int, np.ndarray] = {}
    for L2 in sorted(train["L2"].unique()):
        mask = (train["L2"] == L2).values
        centroids[int(L2)] = Xs_train[mask].mean(axis=0)

    test_mask = kpi["L2"].isna()
    X_test = kpi.loc[test_mask, feat_cols].fillna(1.0).values
    if X_test.size == 0:
        kpi["L2"] = kpi["L2"].astype(int)
        return kpi
    Xs_test = scaler.transform(X_test)
    assigned = []
    for vec in Xs_test:
        d_min, best = float("inf"), 0
        for L2, c in centroids.items():
            d = np.linalg.norm(vec - c)
            if d < d_min:
                d_min, best = d, L2
        assigned.append(best)
    kpi.loc[test_mask, "L2"] = assigned
    kpi["L2"] = kpi["L2"].astype(int)
    return kpi


def build(
    scope: str = "main",
    min_n: int = 30,
    max_total_horses: int | None = None,
    out_dir: Path = OUT_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 既存 L2 マッピング
    uni = pd.read_parquet(ART_DIR / "unified.parquet")
    sid_to_l2: dict[str, int] = {}
    for _, r in uni[uni["entity_type"] == "stallion"].iterrows():
        if int(r["L2"]) >= 0:
            sid_to_l2[str(r["entity_id"])] = int(r["L2"])

    meta: dict[str, Any] = {
        "scope": scope, "min_n": min_n,
        "max_total_horses": max_total_horses, "views": {},
    }
    all_profiles: dict[str, Any] = {}

    for view in ["father", "dam_sire_line", "dam_dam_line"]:
        kpi = build_view_kpi_table(scope, view, min_n=min_n, max_total_horses=max_total_horses)
        if kpi.empty:
            print(f"[L3] view={view} 集計結果が空", flush=True)
            continue
        kpi["stallion_id"] = kpi["stallion_id"].astype(str)
        kpi = _relativize(kpi)
        # all scope: L2 未登録の祖先を centroid 距離で最近接 L2 に割り当て
        if scope == "all":
            kpi = _expand_l2_assignment(kpi, sid_to_l2)
        else:
            kpi["L2"] = kpi["stallion_id"].map(sid_to_l2)
            kpi = kpi.dropna(subset=["L2"])
            kpi["L2"] = kpi["L2"].astype(int)
        kpi = _attach_tags(kpi)

        result, profiles = cluster_view_within_l2(kpi, view)
        kpi_with_l3 = kpi.merge(result, on=["stallion_id", "L2"], how="left")
        out_path = out_dir / f"{view}_l3.parquet"
        kpi_with_l3.to_parquet(out_path, index=False, compression="zstd")
        meta["views"][view] = {
            "n_stallions": int(len(kpi_with_l3)),
            "n_l3_total": int(kpi_with_l3.groupby("L2")["L3"].nunique().sum()),
            "out_path": str(out_path),
        }
        for k, v in profiles.items():
            all_profiles[k] = v
        print(f"[L3] view={view} L3 構造: {dict(kpi_with_l3.groupby('L2')['L3'].nunique())}", flush=True)

    (out_dir / "l3_profiles.json").write_text(
        json.dumps(all_profiles, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    meta["generated_at"] = pd.Timestamp.now().isoformat()
    (out_dir / "l3_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[L3] DONE: {out_dir}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["main", "all", "niche"], default="main")
    parser.add_argument("--min-n", type=int, default=30,
                        help="集計レコード最小数 (これ未満は除外)")
    parser.add_argument("--max-total-horses", type=int, default=None,
                        help="子孫頭数上限 (これ以上の祖先は除外。例: 3000 でニッチ特化)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(scope=args.scope, min_n=args.min_n, max_total_horses=args.max_total_horses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
