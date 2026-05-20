"""
血統メタクラスタリング Web API ロジック

ノートブック `notebooks/pedigree/bloodline_subgroup_analysis.ipynb` §21〜§22 の
クラスタリング結果 (data/research/bloodline_meta_cluster/) を読み込み、
馬名 → 父 → クラスタ判定 → 強み・弱みプロファイル を提供する。

- `analyze_horse(name)`: 馬名から完全な分析結果を返す
- `list_clusters()`: 全 L2 メタクラスタの構成 + プロファイル
- `get_meta()`: アーティファクトのメタ情報 (生成日時、エンティティ数等)
"""
from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "data/page_reference/note_aptitude_race"
PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
IDX_DIR = ROOT / "data/page_reference/pedigree_race_index"
PED_10GEN_DIR = ROOT / "data/local/research/pedigree_10gen"

# 距離区分は父視点 (build_meta_cluster_artifacts.py) と統一:
#   短距離 [0,1400) / マイル [1400,1800) / 中距離 [1800,2400) / 長距離 [2400,9999)
DIST_BINS = [0, 1400, 1800, 2400, 9999]
DIST_LABELS = ["短距離", "マイル", "中距離", "長距離"]
STEEP_VENUES = ["中山", "阪神", "中京"]

# 血統 role の重み（証拠ベース、role_lift_evidence.py の H1〜H4 に基づく）
#   F             (父)   : サンプル最大・std 最小、最も安定 → 主軸
#   MF            (母父)  : F と相関 r=0.04〜0.31 で独立 → 第二の重み
#   FF            (父父)  : 長距離・芝で有意 (Δ=+0.28, p=0.004)
#   maternal_deep (母母父以降): MMF∪MMMF∪MMMMF のゆる制約 union
#                              ポジション問わず深母系に登場する馬全体で lift を推定
#                              推論時は MMF/MMMF/MMMMF 各ポジションの種牡馬を参照して平均化
ROLE_BLEND_WEIGHTS_BASE: dict[str, float] = {
    "F":             0.50,
    "MF":            0.25,
    "FF":            0.15,
    "maternal_deep": 0.15,
}
# 信頼度補正の擬似サンプル数 (n / (n+conf_pseudo_n))
ROLE_CONF_PSEUDO_N = 100

# サブモジュールが書き換える可能性のあるグローバル
_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _condition_label(c: str) -> str:
    """特徴量列名 → 人間に読みやすい日本語ラベル"""
    if c.startswith("win_pace_"):
        parts = c.split("_")
        return f"ラップ {parts[2]} × {parts[3]}"
    if c.startswith("win_v_"):
        return f"競馬場: {c.split('_')[-1]}"
    if c.startswith("win_s_"):
        return f"路面: {c.split('_')[-1]}"
    if c.startswith("win_d_"):
        return f"距離: {c.split('_')[-1]}"
    if c == "win_steep":
        return "急坂コース"
    if c == "win_flat":
        return "平坦コース"
    if c == "win_heavy":
        return "重・不良馬場"
    if c == "win_outer":
        return "外回りコース"
    if c == "win_inner":
        return "内回りコース"
    if c == "win_eb_total":
        return "全体 EB 勝率"
    return c


def _load_artifacts() -> Optional[dict[str, Any]]:
    """アーティファクトを 1 度だけロードしてキャッシュする (スレッドセーフ)。"""
    with _lock:
        if _cache.get("loaded"):
            return _cache
        meta_path = ART_DIR / "meta.json"
        if not meta_path.exists():
            logger.warning(
                "アーティファクトが未生成: %s\n"
                "  -> python -m src.research.pedigree.build_meta_cluster_artifacts を実行してください",
                meta_path,
            )
            return None
        try:
            _cache["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            unified = pd.read_parquet(ART_DIR / "unified.parquet")
            _cache["unified"] = unified
            _cache["unified_by_entity"] = unified.set_index("entity_id")
            _cache["l2_profiles"] = {
                int(k): v for k, v in json.loads((ART_DIR / "l2_profiles.json").read_text(encoding="utf-8")).items()
            }
            _cache["overall"] = json.loads((ART_DIR / "overall_raw.json").read_text(encoding="utf-8"))
            _cache["root_of"] = json.loads((ART_DIR / "non_main_root_of.json").read_text(encoding="utf-8"))
            _cache["nm_labels"] = json.loads((ART_DIR / "non_main_group_labels.json").read_text(encoding="utf-8"))
            _cache["sid_to_main"] = json.loads((ART_DIR / "sid_to_main.json").read_text(encoding="utf-8"))
            _cache["sid_to_name"] = json.loads((ART_DIR / "sid_to_name.json").read_text(encoding="utf-8"))
            _cache["scaler"] = json.loads((ART_DIR / "scaler.json").read_text(encoding="utf-8"))
            l2_names_path = ART_DIR / "l2_names.json"
            _cache["super_groups_meta"] = {}
            if l2_names_path.exists():
                l2_names_data = json.loads(l2_names_path.read_text(encoding="utf-8"))
                if "l2_names" in l2_names_data:
                    _cache["l2_names"] = {
                        int(k): v for k, v in l2_names_data["l2_names"].items()
                    }
                    _cache["super_groups_meta"] = l2_names_data.get("super_groups", {})
                else:
                    _cache["l2_names"] = {
                        int(k): v for k, v in l2_names_data.items()
                    }
            else:
                _cache["l2_names"] = {}
            # L2_fine 関連アーティファクト (新規)
            for fname in ("l2_centroids.json", "l2_positions_2d.json",
                            "l2_super_groups.json", "l2_similarity.json",
                            "ancestor_positions_2d.json", "ancestor_to_l2.json"):
                fpath = ART_DIR / fname
                if fpath.exists():
                    try:
                        _cache[fname.replace(".json", "")] = json.loads(
                            fpath.read_text(encoding="utf-8")
                        )
                    except Exception:
                        _cache[fname.replace(".json", "")] = {}
            knn_data = json.loads((ART_DIR / "knn_data.json").read_text(encoding="utf-8"))
            _cache["knn_data"] = knn_data
            _cache["knn_vec"] = np.asarray([d["vec"] for d in knn_data], dtype=float)

            # role 別 lift プロファイル (新規)
            role_path = ART_DIR / "role_lift_profiles.parquet"
            if role_path.exists():
                role_df = pd.read_parquet(role_path)
                role_df["stallion_id"] = role_df["stallion_id"].astype(str)
                # (stallion_id, role) -> row dict にしてキャッシュ
                role_idx: dict[tuple[str, str], dict[str, Any]] = {}
                for _, r in role_df.iterrows():
                    role_idx[(r["stallion_id"], r["role"])] = r.to_dict()
                _cache["role_lift_df"] = role_df
                _cache["role_lift_idx"] = role_idx
                fb_path = ART_DIR / "role_lift_main_group_fallback.json"
                if fb_path.exists():
                    _cache["role_lift_fallback"] = json.loads(fb_path.read_text(encoding="utf-8"))
                else:
                    _cache["role_lift_fallback"] = {}
                meta_role_path = ART_DIR / "role_lift_meta.json"
                if meta_role_path.exists():
                    _cache["role_lift_meta"] = json.loads(meta_role_path.read_text(encoding="utf-8"))
                else:
                    _cache["role_lift_meta"] = {}
            else:
                logger.warning(
                    "role_lift_profiles.parquet 未生成: %s\n"
                    "  -> python -m src.research.pedigree.build_role_lift_profiles",
                    role_path,
                )
                _cache["role_lift_df"] = None
                _cache["role_lift_idx"] = {}
                _cache["role_lift_fallback"] = {}
                _cache["role_lift_meta"] = {}

            # ペア lift プロファイル (新規, 階層フォールバック用)
            #   2026-05-15: PAIR_INDIV (父個体×母父個体) を廃止し、
            #               PAIR_BMS_ROOT (父個体×母父父系root, 母父系) に置換
            for tag, fname in (
                ("pair_bms_root", "pair_lift_profiles_bms_root.parquet"),
                ("pair_group",    "pair_lift_profiles_group.parquet"),
                ("pair_gxg",      "pair_lift_profiles_gxg.parquet"),
            ):
                p = ART_DIR / fname
                if not p.exists():
                    _cache[f"{tag}_df"] = None
                    _cache[f"{tag}_idx"] = {}
                    continue
                pdf = pd.read_parquet(p)
                if "sire_id" in pdf.columns:
                    pdf["sire_id"] = pdf["sire_id"].astype(str)
                if "bms_id" in pdf.columns:
                    pdf["bms_id"] = pdf["bms_id"].astype(str)
                if "bms_root_id" in pdf.columns:
                    pdf["bms_root_id"] = pdf["bms_root_id"].astype(str)
                _cache[f"{tag}_df"] = pdf
                if tag == "pair_bms_root":
                    idx = {(r["sire_id"], r["bms_root_id"]): r.to_dict() for _, r in pdf.iterrows()}
                elif tag == "pair_group":
                    idx = {(r["sire_id"], r["bms_main"]): r.to_dict() for _, r in pdf.iterrows()}
                else:  # pair_gxg
                    idx = {(r["sire_main"], r["bms_main"]): r.to_dict() for _, r in pdf.iterrows()}
                _cache[f"{tag}_idx"] = idx
            pair_meta_path = ART_DIR / "pair_lift_meta.json"
            _cache["pair_lift_meta"] = (
                json.loads(pair_meta_path.read_text(encoding="utf-8"))
                if pair_meta_path.exists() else {}
            )

            # horse_id → bms_root_id, bms_root_name のマップ (pair_bms_root 検索用)
            try:
                hbms_p = IDX_DIR / "horse_bms.parquet"
                if hbms_p.exists():
                    hbms = pd.read_parquet(
                        hbms_p,
                        columns=["horse_id", "bms_root_id", "bms_root_name"],
                    )
                    hbms["horse_id"] = hbms["horse_id"].astype(str)
                    hbms["bms_root_id"] = (
                        hbms["bms_root_id"].astype(str).fillna("").replace({"None": ""})
                    )
                    hbms["bms_root_name"] = (
                        hbms["bms_root_name"].astype(str).fillna("").replace({"None": ""})
                    )
                    _cache["horse_to_bms_root_id"] = dict(zip(hbms["horse_id"], hbms["bms_root_id"]))
                    _cache["horse_to_bms_root_name"] = dict(zip(hbms["horse_id"], hbms["bms_root_name"]))
                else:
                    logger.warning("horse_bms.parquet が見つからない: %s", hbms_p)
                    _cache["horse_to_bms_root_id"] = {}
                    _cache["horse_to_bms_root_name"] = {}
            except Exception as e:
                logger.warning("horse_bms.parquet ロード失敗: %s", e)
                _cache["horse_to_bms_root_id"] = {}
                _cache["horse_to_bms_root_name"] = {}

            race = pd.read_parquet(
                IDX_DIR / "race_result_slim.parquet",
                columns=["horse_id", "horse_name", "date"],
            )
            race["horse_id"] = race["horse_id"].astype(str)
            race = (
                race.dropna(subset=["horse_name"])
                .sort_values("date")
                .drop_duplicates("horse_id", keep="last")
            )
            name_to_id: dict[str, str] = {}
            id_to_name: dict[str, str] = {}
            for hn, hid in zip(race["horse_name"].astype(str).str.strip(), race["horse_id"]):
                if hn and hn not in name_to_id:
                    name_to_id[hn] = hid
                id_to_name[hid] = hn
            _cache["name_to_id"] = name_to_id
            _cache["id_to_name"] = id_to_name

            _cache["loaded"] = True
            logger.info(
                "bloodline_meta_cluster ready: %d entities, L2=%d, %d horses indexed",
                _cache["meta"].get("n_entities", 0),
                _cache["meta"].get("n_l2", 0),
                len(name_to_id),
            )
            return _cache
        except Exception as e:
            logger.error("アーティファクトロード失敗: %s", e, exc_info=True)
            _cache.clear()
            return None


def reload_artifacts() -> bool:
    """アーティファクトを再ロード (notebook 再実行後の反映用)。"""
    with _lock:
        _cache.clear()
    return _load_artifacts() is not None


def is_ready() -> bool:
    return _load_artifacts() is not None


# ── 検索ヘルパー ────────────────────────────────────────────────


def find_horse_id(name: str) -> tuple[Optional[str], Optional[str], list[str]]:
    """馬名 → (horse_id, matched_name, candidates)。完全一致 → 部分一致 の順。"""
    art = _load_artifacts()
    if art is None:
        return None, None, []
    name = (name or "").strip()
    if not name:
        return None, None, []
    n2i = art["name_to_id"]
    if name in n2i:
        return n2i[name], name, []
    cands = [k for k in n2i if name in k][:8]
    if cands:
        return n2i[cands[0]], cands[0], cands
    return None, None, []


def get_sire_from_ped(horse_id: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """5 代血統 JSON から父の (id, name, dam_sire_name) を返す。"""
    horse_id = str(horse_id)
    p = PED_DIR / horse_id[:4] / f"{horse_id}.json"
    if not p.exists():
        return None, None, None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None
    sire_id, sire_name = None, rec.get("sire")
    for a in rec.get("ancestors", []):
        if a.get("generation") == 1 and a.get("position") == 0:
            sire_id = a.get("horse_id")
            sire_name = a.get("name")
            break
    return sire_id, sire_name, rec.get("dam_sire")


def lookup_stallion(sire_id: Optional[str]) -> Optional[dict[str, Any]]:
    """父 → unified の行 (L1_sub, L2 含む)。非主流は group_id 経由。"""
    art = _load_artifacts()
    if art is None or sire_id is None:
        return None
    sire_id = str(sire_id)
    main_g = art["sid_to_main"].get(sire_id)
    ub = art["unified_by_entity"]

    if sire_id in ub.index:
        row = ub.loc[sire_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {
            "sire_id": sire_id,
            "sire_name": art["sid_to_name"].get(sire_id, str(row["entity_label"])),
            "main_group": main_g or str(row["main_group"]),
            "entity_id": sire_id,
            "entity_label": str(row["entity_label"]),
            "L1_sub": int(row["L1_sub"]),
            "L2": int(row["L2"]) if int(row["L2"]) >= 0 else None,
            "n_horses": int(row["n_horses"]),
            "lookup_kind": "direct_main",
        }

    root = art["root_of"].get(sire_id)
    if root and root in ub.index:
        row = ub.loc[root]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {
            "sire_id": sire_id,
            "sire_name": art["sid_to_name"].get(sire_id, "?"),
            "main_group": "非主流",
            "entity_id": str(root),
            "entity_label": str(row["entity_label"]),
            "L1_sub": int(row["L1_sub"]),
            "L2": int(row["L2"]) if int(row["L2"]) >= 0 else None,
            "n_horses": int(row["n_horses"]),
            "lookup_kind": "direct_nonmain_group",
        }
    return None


def assign_unknown_stallion(sire_id: str) -> Optional[dict[str, Any]]:
    """unified に無い父について、レース実績から特徴量を作って KNN で最近傍 L2 を割り当て。

    ※ 重い処理 (race / cats を読み込む) なので頻度低い想定。"""
    art = _load_artifacts()
    if art is None:
        return None
    sire_id = str(sire_id)
    try:
        cats = pd.read_parquet(
            IDX_DIR / "horse_pedigree_cats.parquet",
            columns=["horse_id", "cat", "gen", "stallion_id", "stallion_name"],
        )
        cats["stallion_id"] = cats["stallion_id"].astype(str)
        cats["horse_id"] = cats["horse_id"].astype(str)
        direct = cats[(cats["cat"] == 1) & (cats["gen"] == 1) & (cats["stallion_id"] == sire_id)]
        if direct.empty:
            return None

        race = pd.read_parquet(IDX_DIR / "race_result_slim.parquet")
        race["horse_id"] = race["horse_id"].astype(str)
        race["win"] = (race["finish_position"] == 1).astype(int)
        race = race[race["finish_position"] > 0]
        rf = race.merge(direct[["horse_id"]], on="horse_id", how="inner")
        if len(rf) < 5:
            return None

        rf["dist_cat"] = pd.cut(rf["distance"], bins=DIST_BINS, labels=DIST_LABELS, right=False)
        rf["is_steep"] = rf["venue"].astype(str).isin(STEEP_VENUES)
        rf["is_heavy"] = rf["track_condition"].astype(str).isin(["重", "不良"])
        # 内/外回り (course_profiles + 公知 OVERRIDE)
        try:
            from src.research.pedigree.build_meta_cluster_artifacts import _attach_course_type
            rf = _attach_course_type(rf)
            rf["is_outer"] = rf["course_type"] == "外"
            rf["is_inner"] = rf["course_type"] == "内"
        except Exception:
            rf["is_outer"] = False
            rf["is_inner"] = False

        eb_pn = float(art["meta"].get("eb_prior_n", 30))
        global_p = float(art["meta"].get("global_win", rf["win"].mean()))

        def _eb(w: float, n: int) -> float:
            if n < 3:
                return float("nan")
            return (w + eb_pn * global_p) / (n + eb_pn)

        cols: list[str] = list(art["scaler"]["cols"])
        feat_row: dict[str, float] = {}
        for c in cols:
            if c.startswith("win_pace_"):
                feat_row[c] = float("nan")  # ラップは欠損 (中央値で補完)
                continue
            if c.startswith("win_v_"):
                v = c.split("_")[-1]
                sub = rf[rf["venue"].astype(str) == v]
            elif c.startswith("win_s_"):
                s = c.split("_")[-1]
                sub = rf[rf["surface"].astype(str) == s]
            elif c.startswith("win_d_"):
                d = c.split("_")[-1]
                sub = rf[rf["dist_cat"].astype(str) == d]
            elif c == "win_steep":
                sub = rf[rf["is_steep"]]
            elif c == "win_flat":
                sub = rf[~rf["is_steep"]]
            elif c == "win_heavy":
                sub = rf[rf["is_heavy"]]
            elif c == "win_outer":
                sub = rf[rf.get("is_outer", pd.Series([False] * len(rf)))]
            elif c == "win_inner":
                sub = rf[rf.get("is_inner", pd.Series([False] * len(rf)))]
            else:
                sub = rf
            feat_row[c] = _eb(float(sub["win"].sum()), int(len(sub)))

        overall = pd.Series(art["overall"])
        row = pd.Series(feat_row).reindex(cols)
        centered = (row - overall.reindex(cols)).fillna(0).astype(float).values
        sc_mean = np.asarray(art["scaler"]["mean"], dtype=float)
        sc_scale = np.asarray(art["scaler"]["scale"], dtype=float)
        sc_scale = np.where(sc_scale == 0, 1.0, sc_scale)
        Xq = ((centered - sc_mean) / sc_scale).reshape(1, -1)

        knn_vec = art["knn_vec"]
        dists = np.linalg.norm(knn_vec - Xq, axis=1)
        top3 = np.argsort(dists)[:3]
        L2_votes: list[int] = []
        neighbors: list[dict[str, Any]] = []
        for idx in top3:
            d = art["knn_data"][int(idx)]
            if d["L2"] >= 0:
                L2_votes.append(int(d["L2"]))
                neighbors.append({
                    "main_group": d["main_group"],
                    "entity_label": d["entity_label"],
                    "L2": int(d["L2"]),
                    "dist": float(dists[idx]),
                })
        if not L2_votes:
            return None
        L2_pred = max(set(L2_votes), key=L2_votes.count)
        return {
            "sire_id": sire_id,
            "sire_name": art["sid_to_name"].get(sire_id, "?"),
            "main_group": art["sid_to_main"].get(sire_id, "?"),
            "entity_id": None,
            "entity_label": art["sid_to_name"].get(sire_id, sire_id),
            "L1_sub": None,
            "L2": int(L2_pred),
            "n_horses": int(rf["horse_id"].nunique()),
            "neighbors": neighbors,
            "lookup_kind": "knn_fallback",
        }
    except Exception as e:
        logger.error("KNN フォールバック失敗 (sire_id=%s): %s", sire_id, e, exc_info=True)
        return None


# ── 公開 API ────────────────────────────────────────────────


def get_l2_meta(L2: Optional[int]) -> Optional[dict[str, Any]]:
    """L2 の名称・サブタイトル・色・特徴量メタを返す。未定義 / -1 / None なら None。"""
    if L2 is None or L2 < 0:
        return None
    art = _load_artifacts()
    if art is None:
        return None
    names: dict[int, dict[str, Any]] = art.get("l2_names", {}) or {}
    info = names.get(int(L2))
    if not info:
        return {"L2": int(L2), "name": f"L2={L2}", "subtitle": "", "color": "#888"}
    return {
        "L2": int(L2),
        "name": info.get("name", f"L2={L2}"),
        "subtitle": info.get("subtitle", ""),
        "color": info.get("color", "#888"),
        "icon": info.get("icon", "●"),
        "highlights": info.get("highlights", []),
        "weaknesses": info.get("weaknesses", []),
        "rationale": info.get("rationale", ""),
    }


def _profile_summary(L2: int, art: dict[str, Any], top_n: int = 7) -> Optional[dict[str, Any]]:
    """L2 クラスタの平均プロファイル要約 — 2 軸の指標を提供する。

    値の正確な定義 (build_l2_fine_clusters.py 参照):
      art["l2_profiles"][L2][cond] = mean_i[(stallion_i の cond 勝率) / (stallion_i の通算勝率)] - 1.0

    ★ つまり「個体差(=種牡馬の絶対力)を消去した、純粋な条件適性のクラスタ平均」
    ┌─────────────────────────────────────────────────────────────────┐
    │ 指標 #1: relative_self_pct   (個体内 相対適性)                   │
    │   各種牡馬が「自身の通算勝率」と比べて、その条件でどれだけ        │
    │   上振れ/下振れするかの平均。                                    │
    │   →「血統がもつ条件選好」を示す。予測時は個体ベースライン補正に。│
    │                                                                  │
    │ 指標 #2: vs_overall_pct       (vs 全体平均)                     │
    │   L2 平均勝率 ÷ 全体平均勝率 − 1。                              │
    │   →「血統選びの絶対的優劣」を示す。予測時はグループ強さの目安に。│
    └─────────────────────────────────────────────────────────────────┘
    """
    if L2 not in art["l2_profiles"]:
        return None
    prof = art["l2_profiles"][L2]

    # ── 指標 #2 (vs 全体平均) を実時間で計算 ──
    overall = art.get("overall") or {}
    unified = art.get("unified")
    vs_overall_map: dict[str, float] = {}
    if unified is not None and len(unified):
        try:
            sub = unified[(unified["entity_type"] == "stallion") & (unified["L2"] == L2)]
            if len(sub) > 0:
                for c in prof.keys():
                    base = overall.get(c)
                    if base is None or base <= 0:
                        continue
                    l2_mean = float(sub[c].mean()) if c in sub.columns else None
                    if l2_mean is None or l2_mean <= 0:
                        continue
                    vs_overall_map[c] = (l2_mean / base - 1.0)
        except Exception:
            pass

    sorted_prof = sorted(prof.items(), key=lambda x: x[1])
    weak = sorted_prof[:top_n]
    strong = sorted_prof[-top_n:][::-1]

    def _entry(c: str, v: float) -> dict[str, Any]:
        rel_self_pct = float(v) * 100.0  # = (rel.mean() - 1.0) * 100
        vs_overall = vs_overall_map.get(c)
        return {
            "cond": c,
            "label": _condition_label(c),
            # ── 指標 #1: 個体内 相対適性 ──
            "lift": float(v) + 1.0,                      # 真の lift (= rel.mean())
            "rel_self_pct": rel_self_pct,                # ★ メイン指標 (純粋な条件適性)
            # ── 指標 #2: vs 全体平均 (予測時のグループ絶対強さ) ──
            "vs_overall_pct": (vs_overall * 100.0) if vs_overall is not None else None,
            # ── レガシー互換 (旧 UI 用、内部的には rel_self_pct と同値) ──
            "lift_pct": rel_self_pct,
            "dev_pct": rel_self_pct,
        }

    return {
        "value_kind": "rel_self_lift_minus_1",
        "label_unit": "個体内相対適性 (各種牡馬の自身通算勝率比のL2平均, % : 0=自身平均通り)",
        "metric_definitions": {
            "rel_self_pct": {
                "name": "個体内 相対適性",
                "definition": "各種牡馬が自身の通算勝率と比べて、その条件でどれだけ上振れ/下振れするかの L2 クラスタ平均。",
                "use_for": "個体ベースライン勝率への補正係数。血統の純粋な条件選好を抽出。",
                "unit": "%",
            },
            "vs_overall_pct": {
                "name": "vs 全体平均",
                "definition": "L2 クラスタの条件別平均勝率 ÷ 全体平均勝率 - 1。",
                "use_for": "血統グループの絶対的な強さ。同条件で他血統と比較した優劣。",
                "unit": "%",
            },
        },
        "strengths": [_entry(c, v) for c, v in strong],
        "weaknesses": [_entry(c, v) for c, v in weak],
        "all": {c: float(v) for c, v in prof.items()},
    }


# ────────────────────────────────────────────────────
# role 別プロファイル生成
# ────────────────────────────────────────────────────

def _role_lift_for(stallion_id: Optional[str], role: str, art: dict[str, Any]) -> Optional[dict[str, Any]]:
    """role_lift_idx から (stallion_id, role) の lift プロファイルを返す。

    返却: {"lift": {cond: lift}, "n_records": int, "n_horses": int, "eb_total": float,
           "stallion_id": str, "stallion_name": str, "main_group": str|None}
    """
    if stallion_id is None:
        return None
    idx: dict[tuple[str, str], dict[str, Any]] = art.get("role_lift_idx") or {}
    row = idx.get((str(stallion_id), role))
    if row is None:
        return None
    lift = {
        k.replace("lift_", "", 1): float(v)
        for k, v in row.items()
        if isinstance(k, str) and k.startswith("lift_") and pd.notna(v)
    }
    n_per_cond = {
        k.replace("n_", "", 1): int(v)
        for k, v in row.items()
        if isinstance(k, str) and k.startswith("n_win_") and pd.notna(v)
    }
    return {
        "stallion_id": str(row.get("stallion_id")),
        "stallion_name": row.get("stallion_name"),
        "main_group": row.get("main_group"),
        "n_horses": int(row.get("n_horses", 0)),
        "n_records": int(row.get("n_records", 0)),
        "eb_total": float(row.get("eb_total", 0.0)),
        "lift": lift,
        "n_per_cond": n_per_cond,
    }


def _fallback_main_group_lift(main_group: Optional[str], role: str, art: dict[str, Any]) -> Optional[dict[str, float]]:
    """role × main_group の平均 lift にフォールバック。サンプル少なすぎる場合 None。"""
    if main_group is None:
        return None
    fb: dict[str, dict[str, dict[str, float]]] = art.get("role_lift_fallback") or {}
    g = fb.get(role, {}).get(str(main_group))
    if not g:
        return None
    return {k.replace("lift_", "", 1): float(v) for k, v in g.items()
            if isinstance(k, str) and k.startswith("lift_")}


def _resolve_horse_role_stallions(horse_id: str) -> dict[str, Optional[dict[str, Any]]]:
    """horse_id → {role: {stallion_id, stallion_name, source}} を 5gen JSON ベースで構築。

    role        gen  pos   path        formula: pos = 2^gen - 2 for M^(gen-1)F
    ---------  ----  ----  ----------  -------
    F            1     0   F
    FF           2     0   FF
    MF           2     2   MF          2^2-2 = 2
    MMF          3     6   MMF         2^3-2 = 6
    MMMF         4    14   MMMF        2^4-2 = 14
    MMMMF        5    30   MMMMF       2^5-2 = 30
    """
    horse_id = str(horse_id)
    p = PED_DIR / horse_id[:4] / f"{horse_id}.json"
    all_roles = ("F", "FF", "MF", "MMF", "MMMF", "MMMMF")
    out: dict[str, Optional[dict[str, Any]]] = {r: None for r in all_roles}
    if not p.exists():
        return out
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    by_pos: dict[tuple[int, int], dict[str, Any]] = {}
    for a in rec.get("ancestors", []) or []:
        try:
            g = int(a.get("generation"))
            pos = int(a.get("position"))
        except (TypeError, ValueError):
            continue
        by_pos[(g, pos)] = a

    POSITIONS = {
        "F":     (1,  0),   # 父
        "FF":    (2,  0),   # 父父
        "MF":    (2,  2),   # 母父
        "MMF":   (3,  6),   # 母母父
        "MMMF":  (4, 14),   # 母母母父  (2^4-2)
        "MMMMF": (5, 30),   # 母母母母父 (2^5-2)
    }
    for role, key in POSITIONS.items():
        anc = by_pos.get(key)
        if anc:
            out[role] = {
                "stallion_id": anc.get("horse_id"),
                "stallion_name": anc.get("name"),
                "source": "ped_5gen",
            }
    return out


def _merge_lift_dicts_weighted(
    entries: list[tuple[float, dict[str, float]]]
) -> dict[str, float]:
    """[(weight, lift_dict), ...] → 重み付き平均 lift を返す。"""
    all_k: set[str] = set()
    for _, ld in entries:
        all_k.update(ld.keys())
    merged: dict[str, float] = {}
    for k in all_k:
        num = 0.0
        den = 0.0
        for w, ld in entries:
            if k in ld:
                num += w * ld[k]
                den += w
        if den > 0:
            merged[k] = num / den
    return merged


def compute_role_blended_profile(
    horse_id: str,
    *,
    art: Optional[dict[str, Any]] = None,
    top_n: int = 7,
) -> Optional[dict[str, Any]]:
    """horse_id について役割別 lift の blended プロファイルを返す。

    maternal_deep は MMF/MMMF/MMMMF の 3ポジション種牡馬を全て参照し、
    n_records 加重平均した lift を 1ロールとして扱う。
    """
    if art is None:
        art = _load_artifacts()
    if art is None:
        return None
    role_stallions = _resolve_horse_role_stallions(horse_id)

    # 各 role の lift とサンプル数を集める
    role_profiles: dict[str, dict[str, Any]] = {}
    diag_missing: list[str] = []
    diag_fallback: list[str] = []

    # ── maternal_deep: MMF/MMMF/MMMMF ポジションを union ────────────────
    _DEEP_POSITIONS = ("MMF", "MMMF", "MMMMF")
    _deep_indiv: list[tuple[int, dict[str, float]]] = []   # (n_records, lift)
    _deep_sids: list[str] = []
    _deep_snames: list[str] = []
    for pos_role in _DEEP_POSITIONS:
        info = role_stallions.get(pos_role)
        if not info:
            continue
        sid_i = info.get("stallion_id")
        prof_i = _role_lift_for(sid_i, "maternal_deep", art)
        if prof_i and prof_i.get("lift"):
            _deep_indiv.append((max(prof_i["n_records"], 1), prof_i["lift"]))
            if sid_i:
                _deep_sids.append(str(sid_i))
            _deep_snames.append(info.get("stallion_name") or "")
    if _deep_indiv:
        total_n = sum(n for n, _ in _deep_indiv)
        merged_lift = _merge_lift_dicts_weighted(
            [(n / total_n, ld) for n, ld in _deep_indiv]
        )
        role_profiles["maternal_deep"] = {
            "stallion_id": ",".join(_deep_sids),
            "stallion_name": ",".join(s for s in _deep_snames if s),
            "main_group": None,
            "n_records": total_n,
            "n_horses": 0,
            "eb_total": None,
            "lift": merged_lift,
            "source": "individual",
        }
    else:
        # fallback: 最初に見つかった deep 祖先の main_group で代替
        sid_to_main = art.get("sid_to_main", {}) or {}
        _fb_lifts: list[dict[str, float]] = []
        for pos_role in _DEEP_POSITIONS:
            info = role_stallions.get(pos_role)
            sid_i = info.get("stallion_id") if info else None
            main_g = sid_to_main.get(str(sid_i)) if sid_i else None
            fb = _fallback_main_group_lift(main_g, "maternal_deep", art)
            if fb:
                _fb_lifts.append(fb)
        if _fb_lifts:
            merged_fb = _merge_lift_dicts_weighted(
                [(1.0, fb) for fb in _fb_lifts]
            )
            role_profiles["maternal_deep"] = {
                "stallion_id": None, "stallion_name": None,
                "main_group": None, "n_records": 0, "n_horses": 0,
                "eb_total": None, "lift": merged_fb,
                "source": "main_group_fallback",
            }
            diag_fallback.append("maternal_deep")
        else:
            diag_missing.append("maternal_deep")

    # ── F / MF / FF: 通常の単一種牡馬ルックアップ ──────────────────────
    for role in ("F", "MF", "FF"):
        info = role_stallions.get(role)
        sid = info.get("stallion_id") if info else None
        sname = info.get("stallion_name") if info else None
        prof = _role_lift_for(sid, role, art)
        if prof is not None and prof["lift"]:
            role_profiles[role] = {
                "stallion_id": sid,
                "stallion_name": sname or prof.get("stallion_name"),
                "main_group": prof.get("main_group"),
                "n_records": prof["n_records"],
                "n_horses": prof["n_horses"],
                "eb_total": prof["eb_total"],
                "lift": prof["lift"],
                "source": "individual",
            }
            continue
        # fallback: main_group 平均
        sid_to_main = art.get("sid_to_main", {}) or {}
        main_g = sid_to_main.get(str(sid)) if sid else None
        fb_lift = _fallback_main_group_lift(main_g, role, art)
        if fb_lift:
            role_profiles[role] = {
                "stallion_id": sid,
                "stallion_name": sname,
                "main_group": main_g,
                "n_records": 0,
                "n_horses": 0,
                "eb_total": None,
                "lift": fb_lift,
                "source": "main_group_fallback",
            }
            diag_fallback.append(role)
            continue
        diag_missing.append(role)

    if not role_profiles:
        return None

    # 信頼度 conf = n / (n + ROLE_CONF_PSEUDO_N), fallback は固定 conf=0.30
    weights_eff: dict[str, float] = {}
    for role, prof in role_profiles.items():
        base_w = ROLE_BLEND_WEIGHTS_BASE.get(role, 0.0)
        if prof["source"] == "individual":
            n = max(prof.get("n_records", 0), 0)
            conf = n / (n + ROLE_CONF_PSEUDO_N)
        else:
            conf = 0.30
        weights_eff[role] = base_w * conf

    w_sum = sum(weights_eff.values())
    if w_sum <= 0:
        return None
    weights_norm = {k: v / w_sum for k, v in weights_eff.items()}

    # 条件別ブレンド
    all_conds: set[str] = set()
    for prof in role_profiles.values():
        all_conds.update(prof["lift"].keys())
    blended_lift: dict[str, float] = {}
    for c in all_conds:
        num = 0.0
        den = 0.0
        for role, prof in role_profiles.items():
            if c in prof["lift"]:
                w = weights_norm[role]
                num += w * prof["lift"][c]
                den += w
        if den > 0:
            blended_lift[c] = num / den

    # 強み・弱み (lift から +/- に並べる)
    sorted_blend = sorted(blended_lift.items(), key=lambda x: x[1])
    weak = sorted_blend[:top_n]
    strong = sorted_blend[-top_n:][::-1]

    def _entry(c: str, v: float) -> dict[str, Any]:
        return {
            "cond": c,
            "label": _condition_label(c),
            "lift": float(v),
            "lift_pct": (float(v) - 1.0) * 100.0,
        }

    # role 表示用に role 別の strengths/weaknesses も付与
    for role, prof in role_profiles.items():
        sp = sorted(prof["lift"].items(), key=lambda x: x[1])
        prof["strengths"] = [_entry(c, v) for c, v in sp[-5:][::-1]]
        prof["weaknesses"] = [_entry(c, v) for c, v in sp[:5]]
        prof["weight_base"] = ROLE_BLEND_WEIGHTS_BASE.get(role, 0.0)
        prof["weight_effective"] = weights_norm[role]

    return {
        "horse_id": str(horse_id),
        "role_stallions": {r: role_stallions.get(r) for r in ROLE_BLEND_WEIGHTS_BASE},
        "role_profiles": role_profiles,
        "blended_lift": blended_lift,
        "blended_strengths": [_entry(c, v) for c, v in strong],
        "blended_weaknesses": [_entry(c, v) for c, v in weak],
        "weights_used": weights_norm,
        "weights_base": ROLE_BLEND_WEIGHTS_BASE,
        "diagnostics": {
            "missing_roles": diag_missing,
            "fallback_roles": diag_fallback,
            "active_roles": list(role_profiles.keys()),
        },
        "value_kind": "lift",
        "label_unit": "lift (1.0=平均、>1で得意)",
    }


# ────────────────────────────────────────────────────────────────────────
# ペア lift (父個体 × 母父個体 / 系統) と階層フォールバック
# ────────────────────────────────────────────────────────────────────────

# 父系プロファイル: F + FF
PATERNAL_VIEW_WEIGHTS_BASE: dict[str, float] = {"F": 0.70, "FF": 0.30}
# 母系プロファイル: MF + MMF (+ deeper if available)
MATERNAL_VIEW_WEIGHTS_BASE: dict[str, float] = {"MF": 0.55, "MMF": 0.45}


def _row_lift(row: dict[str, Any]) -> dict[str, float]:
    """parquet 行 dict から lift_<cond> を抽出して {cond: lift} に変換。"""
    out: dict[str, float] = {}
    for k, v in row.items():
        if isinstance(k, str) and k.startswith("lift_") and v is not None and pd.notna(v):
            out[k.replace("lift_", "", 1)] = float(v)
    return out


def _resolve_bms_root_for_horse(
    horse_id: Optional[str], art: dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
    """horse_id → (bms_root_id, bms_root_name) を解決する。

    horse_bms.parquet のキャッシュを使う。未登録 / 海外馬等で root が無い場合は (None, None)。
    """
    if not horse_id:
        return None, None
    hid = str(horse_id)
    rid = (art.get("horse_to_bms_root_id") or {}).get(hid) or None
    rname = (art.get("horse_to_bms_root_name") or {}).get(hid) or None
    if not rid:
        return None, None
    return rid, rname


def _pair_lift_bms_root(
    sire_id: Optional[str], bms_root_id: Optional[str], art: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """父個体 × 母父父系root (=母父系) ペア の lift を返す。

    旧 _pair_lift_indiv (父個体×母父個体) を 2026-05-15 に置き換え。
    母父個体だとペア件数が薄くなる問題に対応するため、母父の '父系ルート種牡馬'
    (= 例: ディープインパクト系 / Sunday Silence系 / Sadler's Wells系 等の 227 系統)
    を中間粒度として採用する。
    """
    if not sire_id or not bms_root_id:
        return None
    idx = art.get("pair_bms_root_idx") or {}
    row = idx.get((str(sire_id), str(bms_root_id)))
    if not row:
        return None
    return {
        "kind": "pair_bms_root",
        "lift": _row_lift(row),
        "n_horses": int(row.get("n_horses", 0)),
        "n_records": int(row.get("n_records", 0)),
        "eb_total": float(row.get("eb_total", 0.0)),
        "sire_id": str(row.get("sire_id")),
        "sire_name": row.get("sire_name"),
        "bms_root_id": str(row.get("bms_root_id")),
        "bms_root_name": row.get("bms_root_name"),
        "bms_main": row.get("bms_main"),
    }


def _pair_lift_group(sire_id: Optional[str], bms_main: Optional[str], art: dict[str, Any]) -> Optional[dict[str, Any]]:
    """父個体 × 母父系統 ペア (n>=50) の lift を返す。"""
    if not sire_id or not bms_main:
        return None
    idx = art.get("pair_group_idx") or {}
    row = idx.get((str(sire_id), str(bms_main)))
    if not row:
        return None
    return {
        "kind": "pair_group",
        "lift": _row_lift(row),
        "n_horses": int(row.get("n_horses", 0)),
        "n_records": int(row.get("n_records", 0)),
        "eb_total": float(row.get("eb_total", 0.0)),
        "sire_id": str(row.get("sire_id")),
        "sire_name": row.get("sire_name"),
        "bms_main": str(row.get("bms_main")),
    }


def _pair_lift_gxg(sire_main: Optional[str], bms_main: Optional[str], art: dict[str, Any]) -> Optional[dict[str, Any]]:
    """父系統 × 母父系統 (5×5) の lift を返す (フォールバック)。"""
    if not sire_main or not bms_main:
        return None
    idx = art.get("pair_gxg_idx") or {}
    row = idx.get((str(sire_main), str(bms_main)))
    if not row:
        return None
    return {
        "kind": "pair_gxg",
        "lift": _row_lift(row),
        "n_horses": int(row.get("n_horses", 0)),
        "n_records": int(row.get("n_records", 0)),
        "eb_total": float(row.get("eb_total", 0.0)),
        "sire_main": str(row.get("sire_main")),
        "bms_main": str(row.get("bms_main")),
    }


# ────────────────────────────────────────────────────────────────────────
# 統合 prior の改善 v2: 複数 layer の加重ブレンド + 信頼度 shrinkage
# ────────────────────────────────────────────────────────────────────────
#
# v1 (階層フォールバック単一採用) のバックテスト結果:
#   - キャリブレーションは完璧に単調 (✓)
#   - 旧 pair_indiv (corr 0.270) と role_blend (corr 0.294) は強いが
#     pair_group (corr 0.206) は精度が低い
#   - 方向性一致率は 53.8% でランダムよりは上だが改善余地あり
#
# 2026-05-15 改訂:
#   - 旧 pair_indiv (父個体×母父個体) はサンプル件数 442 ペアと薄く、ユーザー要件
#     に基づき pair_bms_root (父個体×母父父系root, 1629ペア n>=5) に置換。
#   - 母父系 (= 母父の "父系ルート種牡馬") 227 系統で集計するため、5大系統より細かく、
#     母父個体より広いサンプルを得られる。
#
# v2 → v3 設計 (グリッド検索 v3d 採用):
#   1. 利用可能な layer を信頼度ベースで加重ブレンド
#   2. pair_bms_root を最優先、pair_gxg は他 layer が全く無い時の純フォールバックに変更
#   3. shrinkage: ブレンド後の lift を 1.0 方向にやや圧縮 (自信過剰の抑制)
#
# 信頼度: conf = n_records / (n_records + LAYER_PSEUDO_N[layer])
# 採用重み: w_layer = base_weight[layer] * conf
# 最終 lift: shrunk = 1.0 + (raw_lift - 1.0) * SHRINK_RATIO
#
# グリッド検索結果 (backtest_prior_v3_grid.py 実行時):
#   v3d: wcorr=+0.2205, agree=0.543, dt_corr=+0.153 (v2 比でほぼ全指標改善)

LAYER_BASE_WEIGHT = {
    "pair_bms_root": 0.60, # 父個体 × 母父系 (227系統) ← メイン軸
    "pair_group":    0.25, # 父個体 × 母父5大系統     ← 中優先
    "role_blend":    0.30, # F+MF 役割別 lift          ← 中優先
    # pair_gxg: 加重ブレンドからは除外。他 layer が無い時の純フォールバックとして使う
}
LAYER_PSEUDO_N = {
    "pair_bms_root": 100,
    "pair_group":    200,
    "role_blend":    100,
    "pair_gxg":      500,
}
SHRINK_RATIO = 0.85  # ブレンド後の lift を 1.0 方向に縮める係数 (自信過剰抑制)


def compute_blended_prior_v2(
    horse_id: str,
    *,
    art: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """改善版 prior: 利用可能な全 layer を信頼度で加重ブレンド + shrinkage 適用。

    返却:
      {
        "lift": {cond: blended_shrunk_lift},
        "raw_lift": {cond: blended_raw_lift},
        "components": {layer: {kind, n_records, weight_eff, lift: {...}}, ...},
        "weights_used": {layer: w_norm},
        "shrink_ratio": SHRINK_RATIO,
      }
    """
    if art is None:
        art = _load_artifacts()
    if art is None:
        return None

    role_blend_obj = compute_role_blended_profile(horse_id, art=art)
    if role_blend_obj is None:
        return None
    role_stallions = role_blend_obj["role_stallions"]
    sire_info = role_stallions.get("F") or {}
    mf_info = role_stallions.get("MF") or {}
    sire_id = sire_info.get("stallion_id")
    bms_id = mf_info.get("stallion_id")
    sid_to_main = art.get("sid_to_main", {}) or {}
    sire_main = sid_to_main.get(str(sire_id)) if sire_id else None
    bms_main = sid_to_main.get(str(bms_id)) if bms_id else None
    # bms_root_id (=母父父系root, 母父系) は horse_bms.parquet から引く
    bms_root_id, bms_root_name = _resolve_bms_root_for_horse(horse_id, art)

    components: dict[str, dict[str, Any]] = {}

    # pair_bms_root  (新メイン軸: 父個体 × 母父系)
    p1 = _pair_lift_bms_root(sire_id, bms_root_id, art)
    if p1 and p1["lift"]:
        n = p1["n_records"]
        conf = n / (n + LAYER_PSEUDO_N["pair_bms_root"])
        components["pair_bms_root"] = {
            "kind": "pair_bms_root", "lift": p1["lift"],
            "n_records": n, "n_horses": p1.get("n_horses", 0),
            "weight_eff": LAYER_BASE_WEIGHT["pair_bms_root"] * conf,
            "meta": {
                "sire_id": p1["sire_id"],
                "bms_root_id": p1["bms_root_id"],
                "bms_root_name": p1.get("bms_root_name"),
            },
        }
    # pair_group
    p2 = _pair_lift_group(sire_id, bms_main, art)
    if p2 and p2["lift"]:
        n = p2["n_records"]
        conf = n / (n + LAYER_PSEUDO_N["pair_group"])
        components["pair_group"] = {
            "kind": "pair_group", "lift": p2["lift"],
            "n_records": n, "n_horses": p2.get("n_horses", 0),
            "weight_eff": LAYER_BASE_WEIGHT["pair_group"] * conf,
            "meta": {"sire_id": p2["sire_id"], "bms_main": p2["bms_main"]},
        }
    # role_blend
    role_lift = role_blend_obj.get("blended_lift") or {}
    if role_lift:
        # role_blend の信頼度は role_profiles の n_records 合計から推定
        n_role = sum((p.get("n_records", 0) or 0) for p in role_blend_obj["role_profiles"].values())
        conf = n_role / (n_role + LAYER_PSEUDO_N["role_blend"])
        components["role_blend"] = {
            "kind": "role_blend", "lift": role_lift,
            "n_records": n_role,
            "weight_eff": LAYER_BASE_WEIGHT["role_blend"] * conf,
            "meta": {"active_roles": role_blend_obj["diagnostics"]["active_roles"]},
        }
    # pair_gxg は加重ブレンドからは除外し、他 layer が無い時の純フォールバックとして使う
    if not components:
        p4 = _pair_lift_gxg(sire_main, bms_main, art)
        if p4 and p4["lift"]:
            n = p4["n_records"]
            components["pair_gxg"] = {
                "kind": "pair_gxg", "lift": p4["lift"],
                "n_records": n,
                "weight_eff": 1.0,
                "meta": {"sire_main": sire_main, "bms_main": bms_main, "fallback_only": True},
            }

    if not components:
        return None
    w_sum = sum(c["weight_eff"] for c in components.values())
    if w_sum <= 0:
        return None
    weights_norm = {k: c["weight_eff"] / w_sum for k, c in components.items()}

    all_conds: set[str] = set()
    for c in components.values():
        all_conds.update(c["lift"].keys())
    raw_blend: dict[str, float] = {}
    for cond in all_conds:
        num = 0.0
        den = 0.0
        for layer, c in components.items():
            if cond in c["lift"]:
                w = weights_norm[layer]
                num += w * c["lift"][cond]
                den += w
        if den > 0:
            raw_blend[cond] = num / den
    shrunk: dict[str, float] = {
        cond: 1.0 + (lift - 1.0) * SHRINK_RATIO for cond, lift in raw_blend.items()
    }
    return {
        "horse_id": str(horse_id),
        "lift": shrunk,
        "raw_lift": raw_blend,
        "components": components,
        "weights_used": weights_norm,
        "shrink_ratio": SHRINK_RATIO,
        "n_layers_used": len(components),
    }


def _build_view_from_roles(
    role_profiles: dict[str, dict[str, Any]],
    view_weights: dict[str, float],
) -> Optional[dict[str, Any]]:
    """role_profiles の中から view_weights に該当する role を取り出して加重平均。

    返却:
      {"lift": {cond: blended}, "weights_used": {role: w_eff}, "active_roles": [..]}
    """
    weights_eff: dict[str, float] = {}
    used_profiles: dict[str, dict[str, Any]] = {}
    for role, base_w in view_weights.items():
        prof = role_profiles.get(role)
        if not prof or not prof.get("lift"):
            continue
        if prof.get("source") == "individual":
            n = max(prof.get("n_records", 0), 0)
            conf = n / (n + ROLE_CONF_PSEUDO_N)
        else:
            conf = 0.30
        weights_eff[role] = base_w * conf
        used_profiles[role] = prof
    w_sum = sum(weights_eff.values())
    if w_sum <= 0:
        return None
    weights_norm = {k: v / w_sum for k, v in weights_eff.items()}

    all_conds: set[str] = set()
    for prof in used_profiles.values():
        all_conds.update(prof["lift"].keys())
    blended: dict[str, float] = {}
    for c in all_conds:
        num = 0.0
        den = 0.0
        for role, prof in used_profiles.items():
            if c in prof["lift"]:
                w = weights_norm[role]
                num += w * prof["lift"][c]
                den += w
        if den > 0:
            blended[c] = num / den
    return {
        "lift": blended,
        "weights_used": weights_norm,
        "active_roles": list(used_profiles.keys()),
    }


def _topn_strengths_weaknesses(lift: dict[str, float], top_n: int = 7) -> tuple[list[dict], list[dict]]:
    items = sorted(lift.items(), key=lambda x: x[1])
    weak = items[:top_n]
    strong = items[-top_n:][::-1]
    def _entry(c, v):
        return {
            "cond": c,
            "label": _condition_label(c),
            "lift": float(v),
            "lift_pct": (float(v) - 1.0) * 100.0,
        }
    return [_entry(c, v) for c, v in strong], [_entry(c, v) for c, v in weak]


def compute_three_view_profile(
    horse_id: str,
    *,
    art: Optional[dict[str, Any]] = None,
    top_n: int = 7,
) -> Optional[dict[str, Any]]:
    """3 ビュー（父系 / 母系 / 統合）プロファイルを構築。

    統合ビューの階層フォールバック (2026-05-15 更新):
      1) 父個体 × 母父系  (pair_bms_root, 母父父系root 227系統, n>=50)
      2) 父個体 × 母父5大系統 (pair_group, n>=50)
      3) F+MF+FF+MMF 重み平均 (compute_role_blended_profile)
      4) 父5大系統 × 母父5大系統 (pair_gxg)
    """
    if art is None:
        art = _load_artifacts()
    if art is None:
        return None

    role_blend = compute_role_blended_profile(horse_id, art=art)
    if role_blend is None:
        return None
    role_profiles = role_blend["role_profiles"]
    role_stallions = role_blend["role_stallions"]

    # 父系・母系の役割視点
    paternal_view = _build_view_from_roles(role_profiles, PATERNAL_VIEW_WEIGHTS_BASE)
    maternal_view = _build_view_from_roles(role_profiles, MATERNAL_VIEW_WEIGHTS_BASE)

    # ── 統合ビュー: 階層フォールバック ──
    sire_info = role_stallions.get("F") or {}
    mf_info = role_stallions.get("MF") or {}
    sire_id = sire_info.get("stallion_id")
    bms_id = mf_info.get("stallion_id")
    sid_to_main = art.get("sid_to_main", {}) or {}
    sire_main = sid_to_main.get(str(sire_id)) if sire_id else None
    bms_main = sid_to_main.get(str(bms_id)) if bms_id else None

    # ── 統合 prior の改善版 v2 (複数 layer 加重ブレンド + shrinkage) ──
    blended_v2 = compute_blended_prior_v2(horse_id, art=art)
    # フォールバックチェーン (診断用): 各 layer の利用可否
    bms_root_id, bms_root_name = _resolve_bms_root_for_horse(horse_id, art)
    p1 = _pair_lift_bms_root(sire_id, bms_root_id, art)
    p2 = _pair_lift_group(sire_id, bms_main, art)
    p4 = _pair_lift_gxg(sire_main, bms_main, art)
    fallback_chain = [
        {"layer": 1, "kind": "pair_bms_root",
         "label_ja": "父個体 × 母父系 (母父父系root)",
         "available": p1 is not None,
         "n_records": p1.get("n_records") if p1 else None,
         "bms_root_id": bms_root_id, "bms_root_name": bms_root_name},
        {"layer": 2, "kind": "pair_group",
         "label_ja": "父個体 × 母父5大系統",
         "available": p2 is not None,
         "n_records": p2.get("n_records") if p2 else None, "bms_main": bms_main},
        {"layer": 3, "kind": "role_blend",
         "label_ja": "父+母父+父父+母母父 役割別 lift 加重平均",
         "available": True,
         "n_records": sum(p.get("n_records", 0) for p in role_profiles.values())},
        {"layer": 4, "kind": "pair_gxg",
         "label_ja": "父5大系統 × 母父5大系統 (最終フォールバック)",
         "available": p4 is not None,
         "n_records": p4.get("n_records") if p4 else None,
         "sire_main": sire_main, "bms_main": bms_main},
    ]
    if blended_v2 is None:
        return None
    integrated_lift = blended_v2["lift"]
    integrated_source = {
        "kind": "blended_v2",
        "components": {k: {"n_records": c["n_records"], "weight_used": blended_v2["weights_used"][k]}
                        for k, c in blended_v2["components"].items()},
        "weights_used": blended_v2["weights_used"],
        "n_layers_used": blended_v2["n_layers_used"],
        "shrink_ratio": blended_v2["shrink_ratio"],
        "raw_lift": blended_v2["raw_lift"],
    }
    int_strengths, int_weaknesses = _topn_strengths_weaknesses(integrated_lift, top_n)

    out = {
        "horse_id": str(horse_id),
        "role_stallions": role_stallions,
        "paternal": None,
        "maternal": None,
        "integrated": {
            "source_layer": integrated_source["kind"],
            "source_meta": integrated_source,
            "lift": integrated_lift,
            "strengths": int_strengths,
            "weaknesses": int_weaknesses,
        },
        "fallback_chain": fallback_chain,
        "value_kind": "lift",
        "label_unit": "lift (1.0=平均、>1で得意)",
    }
    if paternal_view:
        s, w = _topn_strengths_weaknesses(paternal_view["lift"], top_n)
        out["paternal"] = {
            "weights_used": paternal_view["weights_used"],
            "active_roles": paternal_view["active_roles"],
            "lift": paternal_view["lift"],
            "strengths": s,
            "weaknesses": w,
            "description": "父系プロファイル (F=父 + FF=父父 を加重平均)",
        }
    if maternal_view:
        s, w = _topn_strengths_weaknesses(maternal_view["lift"], top_n)
        out["maternal"] = {
            "weights_used": maternal_view["weights_used"],
            "active_roles": maternal_view["active_roles"],
            "lift": maternal_view["lift"],
            "strengths": s,
            "weaknesses": w,
            "description": "母系プロファイル (MF=母父 + MMF=母母父 を加重平均)",
        }
    return out


# ────────────────────────────────────────────────────────────────────────
# 個体実績 vs 血統 prior 比較
# ────────────────────────────────────────────────────────────────────────

def _compute_horse_actual_lift(horse_id: str, art: dict[str, Any]) -> Optional[dict[str, Any]]:
    """対象馬の実出走から条件別 win_rate と本馬全体勝率の比 (個体 lift) を計算。

    pair lift と同じ条件軸 (cond) で出力。サンプル少 (n<3) はスキップ。
    """
    art2 = _load_stats_assets()
    if art2 is None:
        return None
    rec = art2["records"]
    sub = rec[rec["horse_id"].astype(str) == str(horse_id)].copy()
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]
    if len(sub) < 3:
        return None
    n_total = int(len(sub))
    win_total = int((sub["finish_position"] == 1).sum())
    rate_total = win_total / n_total
    if rate_total <= 0:
        return None

    # dist / steep / heavy / 内外回り 派生
    sub = sub.assign(
        dist_cat=pd.cut(sub["distance"], bins=DIST_BINS, labels=DIST_LABELS, right=False).astype(str),
        is_steep=sub["venue"].astype(str).isin(STEEP_VENUES),
        is_heavy=sub["track_condition"].astype(str).isin(["重", "不良"]),
        win=(sub["finish_position"] == 1).astype(int),
    )
    try:
        from src.research.pedigree.build_meta_cluster_artifacts import _attach_course_type
        sub = _attach_course_type(sub)
        sub["is_outer"] = sub["course_type"] == "外"
        sub["is_inner"] = sub["course_type"] == "内"
    except Exception:
        sub["is_outer"] = False
        sub["is_inner"] = False

    cond_filters: dict[str, pd.Series] = {}
    for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]:
        cond_filters[f"win_v_{v}"] = sub["venue"].astype(str) == v
    for s in ["芝","ダート"]:
        cond_filters[f"win_s_{s}"] = sub["surface"].astype(str) == s
    for d in DIST_LABELS:
        cond_filters[f"win_d_{d}"] = sub["dist_cat"] == d
    cond_filters["win_steep"] = sub["is_steep"]
    cond_filters["win_flat"] = ~sub["is_steep"]
    cond_filters["win_heavy"] = sub["is_heavy"]
    cond_filters["win_outer"] = sub["is_outer"]
    cond_filters["win_inner"] = sub["is_inner"]

    out: dict[str, Any] = {
        "n_total": n_total,
        "win_total": win_total,
        "win_rate_total": rate_total,
        "by_cond": {},
    }
    for c, mask in cond_filters.items():
        sc = sub[mask]
        n = int(len(sc))
        w = int(sc["win"].sum())
        if n < 3:
            continue
        rate = w / n
        out["by_cond"][c] = {
            "n": n,
            "win": w,
            "win_rate": rate,
            "individual_lift": rate / rate_total,  # 本馬の全体勝率を分母にした個体 lift
        }
    return out


# ────────────────────────────────────────────────────────────────────────
# 戦績テーブル + 統計的ラベル付け
# ────────────────────────────────────────────────────────────────────────
#
# 各条件 (場所/距離/路面/馬場) について:
#   ・本馬の戦績 (n, w, win_rate)
#   ・血統 prior の期待勝率 (prior_lift × 全体勝率)
#   ・二項検定 (one-sided) で個体勝率が prior と統計的に有意に乖離しているか判定
#   ・Wilson 95% CI で個体勝率の信頼区間
#   ・ラベル付け (6 区分)
#
# ラベル設計:
#   PERSONAL_BREAKTHROUGH (⚡)  個体 >> prior、p<0.05         "個体特異・突出"
#   PERSONAL_UNDERPERFORM (⚠)  個体 << prior、p<0.05         "個体特異・低調"
#   PEDIGREE_OUTPERFORM   (★)  prior 高 & 個体も整合          "血統適性発揮"
#   PEDIGREE_UNDERPERFORM (▽)  prior 低 & 個体も整合 (実証)   "血統通り苦手"
#   ON_PRIOR              (○)  個体 ≈ prior                   "血統通り"
#   INSUFFICIENT_DATA     (—)  n < 5                          "サンプル不足"

LABEL_DEFS = {
    "PERSONAL_BREAKTHROUGH": {
        "icon": "⚡", "color": "purple",
        "ja_short": "個体特異・突出",
        "ja_full": "血統から想定される得意度を統計的に大きく上回った条件。本馬ならではの強み。",
    },
    "PERSONAL_UNDERPERFORM": {
        "icon": "⚠", "color": "orange",
        "ja_short": "個体特異・低調",
        "ja_full": "血統から想定される得意度を統計的に大きく下回った条件。本馬個別の苦手。",
    },
    "PEDIGREE_OUTPERFORM": {
        "icon": "★", "color": "blue",
        "ja_short": "血統適性発揮",
        "ja_full": "血統から得意と想定される条件で、実成績もそれを裏付けた (血統通りの強み)。",
    },
    "PEDIGREE_UNDERPERFORM": {
        "icon": "▽", "color": "gray",
        "ja_short": "血統通り苦手",
        "ja_full": "血統から苦手と想定される条件で、実成績もそれを裏付けた (血統通りの苦手)。",
    },
    "ON_PRIOR": {
        "icon": "○", "color": "green",
        "ja_short": "血統通り",
        "ja_full": "本馬の実成績が、血統から想定される傾向とほぼ同じ。突出も低調も無し。",
    },
    "INSUFFICIENT_DATA": {
        "icon": "—", "color": "muted",
        "ja_short": "サンプル不足",
        "ja_full": "出走数が少ないため統計的判定不可。",
    },
}

# 判定閾値
P_VALUE_THRESHOLD = 0.10      # 二項検定の有意水準 (緩めの 0.10、サンプル少も拾う)
P_VALUE_STRONG = 0.05         # 強いエビデンス
PRIOR_HIGH_THRESHOLD = 1.10   # prior 高判定 (lift >= 1.10)
PRIOR_LOW_THRESHOLD = 0.90    # prior 低判定 (lift <= 0.90)
ON_PRIOR_LIFT_DIFF = 0.15     # |個体 lift - prior lift| < 0.15 → 整合
MIN_N_FOR_TEST = 3            # 検定実施に必要な最小出走数 (n=3 全勝でも p<0.01 なら有意)
TENTATIVE_N = 5               # n<TENTATIVE_N の判定は "暫定" 扱い (CI が広い)


def _wilson_ci(w: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    from math import sqrt
    p = w / n
    z = 1.959963984540054 if conf == 0.95 else 1.6448536269514722
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_pvalue(w: int, n: int, p_expected: float) -> tuple[float, str]:
    """個体勝率が prior 期待値から有意に乖離しているか one-sided 検定。

    観測方向 (over/under) を判定し、その方向の片側 p-value を返す。
    「乖離の方向」自体が結論なので片側検定が妥当。
    返却: (p_value, direction) direction in {"over","under","equal"}
    """
    from scipy.stats import binomtest
    if n <= 0:
        return (1.0, "equal")
    p_expected = max(min(p_expected, 0.999), 0.001)
    obs_rate = w / n
    if obs_rate > p_expected:
        direction = "over"
        res = binomtest(w, n, p_expected, alternative="greater")
    elif obs_rate < p_expected:
        direction = "under"
        res = binomtest(w, n, p_expected, alternative="less")
    else:
        return (1.0, "equal")
    return (float(res.pvalue), direction)


def _classify_pattern(
    n: int, w: int, individual_lift: float, prior_lift: float,
    expected_rate: float, p_value: float, direction: str,
) -> str:
    """6 区分のラベルに分類。"""
    if n < MIN_N_FOR_TEST:
        return "INSUFFICIENT_DATA"
    diff = individual_lift - prior_lift
    if p_value < P_VALUE_THRESHOLD:
        if direction == "over":
            return "PERSONAL_BREAKTHROUGH"
        if direction == "under":
            return "PERSONAL_UNDERPERFORM"
    if abs(diff) < ON_PRIOR_LIFT_DIFF:
        if prior_lift >= PRIOR_HIGH_THRESHOLD:
            return "PEDIGREE_OUTPERFORM"
        if prior_lift <= PRIOR_LOW_THRESHOLD:
            return "PEDIGREE_UNDERPERFORM"
        return "ON_PRIOR"
    if diff > 0 and prior_lift >= PRIOR_HIGH_THRESHOLD:
        return "PEDIGREE_OUTPERFORM"
    if diff < 0 and prior_lift <= PRIOR_LOW_THRESHOLD:
        return "PEDIGREE_UNDERPERFORM"
    return "ON_PRIOR"


# 条件のグルーピング (UI 表示用)
COND_GROUPS = {
    "venue": [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]],
    "surface": [f"win_s_{s}" for s in ["芝","ダート"]],
    "distance": [f"win_d_{d}" for d in DIST_LABELS],
    "topography": ["win_steep","win_flat"],
    "course_shape": ["win_outer","win_inner"],
    "track_condition": ["win_heavy"],
}


def compute_record_with_labels(
    horse_id: str,
    *,
    art: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """本馬の戦績を血統 prior と比較し、各条件に統計的ラベルを付ける。

    返却:
      {
        "summary": {n_total, win_total, win_rate, win_ci},
        "groups": {
          "venue": [{cond, label_jp, n, win, win_rate, win_ci, prior_lift, individual_lift,
                     expected_win_rate, p_value, direction, pattern, pattern_def}, ...],
          "surface": [...], "distance": [...], "topography": [...], "track_condition": [...]
        },
        "notable_personal": [...],   # PERSONAL_BREAKTHROUGH の主要なもの
        "notable_personal_low": [...], # PERSONAL_UNDERPERFORM の主要なもの
        "summary_text": "本馬は ... が血統と異なる個体特性。",
      }
    """
    if art is None:
        art = _load_artifacts()
    if art is None:
        return None

    actual = _compute_horse_actual_lift(horse_id, art)
    three_view = compute_three_view_profile(horse_id, art=art)
    if not actual or not three_view:
        return None
    prior_lift_map = three_view["integrated"]["lift"]
    rate_total = actual["win_rate_total"]

    # 通算
    n_tot = actual["n_total"]; w_tot = actual["win_total"]
    ci_tot = _wilson_ci(w_tot, n_tot)
    summary = {
        "n_total": n_tot, "win_total": w_tot,
        "win_rate_total": rate_total,
        "win_rate_ci_low": ci_tot[0], "win_rate_ci_high": ci_tot[1],
    }

    by_cond_actual = actual["by_cond"]
    groups: dict[str, list[dict[str, Any]]] = {}
    all_cells: list[dict[str, Any]] = []

    for group_name, cond_list in COND_GROUPS.items():
        rows: list[dict[str, Any]] = []
        for cond in cond_list:
            info = by_cond_actual.get(cond)
            n = (info or {}).get("n", 0)
            w = (info or {}).get("win", 0)
            individual_lift = (info or {}).get("individual_lift")
            individual_rate = (info or {}).get("win_rate")
            prior_lift = prior_lift_map.get(cond)
            if prior_lift is None:
                continue  # prior が無ければ表示しない
            expected_rate = prior_lift * rate_total
            ci = _wilson_ci(w, n) if n > 0 else (0.0, 0.0)
            if n >= MIN_N_FOR_TEST and individual_lift is not None:
                p_value, direction = _binom_pvalue(w, n, expected_rate)
                pattern = _classify_pattern(
                    n, w, individual_lift, prior_lift, expected_rate, p_value, direction
                )
            else:
                p_value, direction = (1.0, "equal")
                pattern = "INSUFFICIENT_DATA" if n < MIN_N_FOR_TEST else "ON_PRIOR"
            is_tentative = (n < TENTATIVE_N) and (pattern not in ("INSUFFICIENT_DATA",))
            evidence = (
                "strong" if p_value < P_VALUE_STRONG
                else "moderate" if p_value < P_VALUE_THRESHOLD
                else "weak"
            )
            cell = {
                "cond": cond,
                "label_jp": _condition_label(cond),
                "n": n, "win": w,
                "win_rate": individual_rate,
                "win_rate_ci_low": ci[0], "win_rate_ci_high": ci[1],
                "prior_lift": float(prior_lift),
                "expected_win_rate": float(expected_rate),
                "individual_lift": individual_lift,
                "deviation_lift": (individual_lift - prior_lift) if individual_lift is not None else None,
                "p_value": p_value,
                "direction": direction,
                "evidence_strength": evidence,
                "pattern": pattern,
                "pattern_icon": LABEL_DEFS[pattern]["icon"],
                "pattern_short": LABEL_DEFS[pattern]["ja_short"] + ("(暫定)" if is_tentative else ""),
                "pattern_full": LABEL_DEFS[pattern]["ja_full"]
                                 + (" ※サンプル少 (n<5) のため暫定判定" if is_tentative else ""),
                "is_significant": p_value < P_VALUE_THRESHOLD,
                "is_tentative": is_tentative,
            }
            rows.append(cell)
            all_cells.append(cell)
        groups[group_name] = rows

    notable_personal = sorted(
        [c for c in all_cells if c["pattern"] == "PERSONAL_BREAKTHROUGH"],
        key=lambda c: (c["p_value"], -c["n"]),
    )
    notable_personal_low = sorted(
        [c for c in all_cells if c["pattern"] == "PERSONAL_UNDERPERFORM"],
        key=lambda c: (c["p_value"], -c["n"]),
    )
    notable_pedigree = sorted(
        [c for c in all_cells
         if c["pattern"] == "PEDIGREE_OUTPERFORM" and c["n"] >= MIN_N_FOR_TEST and c["prior_lift"] >= PRIOR_HIGH_THRESHOLD],
        key=lambda c: -c["prior_lift"],
    )
    notable_pedigree_under = sorted(
        [c for c in all_cells
         if c["pattern"] == "PEDIGREE_UNDERPERFORM" and c["n"] >= MIN_N_FOR_TEST and c["prior_lift"] <= PRIOR_LOW_THRESHOLD],
        key=lambda c: c["prior_lift"],
    )

    # サマリーテキスト
    parts: list[str] = []
    if notable_personal:
        labels = "・".join(c["label_jp"] for c in notable_personal[:3])
        parts.append(f"⚡ 個体特異な突出: {labels}")
    if notable_personal_low:
        labels = "・".join(c["label_jp"] for c in notable_personal_low[:3])
        parts.append(f"⚠ 個体特異な低調: {labels}")
    if notable_pedigree:
        labels = "・".join(c["label_jp"] for c in notable_pedigree[:3])
        parts.append(f"★ 血統適性発揮: {labels}")
    summary_text = " / ".join(parts) if parts else (
        "本馬の実成績は、血統から想定される得意・苦手の傾向と概ね一致しています。"
        "統計的に有意な突出・低調 (個体特異性) は検出されませんでした。"
    )

    return {
        "horse_id": str(horse_id),
        "summary": summary,
        "groups": groups,
        "notable_personal": notable_personal,
        "notable_personal_low": notable_personal_low,
        "notable_pedigree": notable_pedigree,
        "notable_pedigree_under": notable_pedigree_under,
        "summary_text": summary_text,
        "label_defs": LABEL_DEFS,
        "thresholds": {
            "p_value": P_VALUE_THRESHOLD,
            "prior_high": PRIOR_HIGH_THRESHOLD,
            "prior_low": PRIOR_LOW_THRESHOLD,
            "on_prior_lift_diff": ON_PRIOR_LIFT_DIFF,
            "min_n": MIN_N_FOR_TEST,
        },
    }


def compute_individual_vs_prior(
    horse_id: str,
    *,
    art: Optional[dict[str, Any]] = None,
    top_n: int = 7,
) -> Optional[dict[str, Any]]:
    """個体実績の条件別 lift と、血統 prior (3ビュー統合) の条件別 lift を比較。

    返却:
      {
        "actual": {n_total, win_total, win_rate, by_cond: {cond: {n, win, win_rate, individual_lift}}},
        "prior":  {cond: prior_lift},     # 統合 prior の lift
        "deviations": [{cond, label, prior_lift, actual_lift, n_actual, deviation_pct}],
        "top_overperform": [...],         # 個体が prior を上回ったコンディション
        "top_underperform": [...],
      }
    """
    if art is None:
        art = _load_artifacts()
    if art is None:
        return None
    actual = _compute_horse_actual_lift(horse_id, art)
    three_view = compute_three_view_profile(horse_id, art=art, top_n=top_n)
    if not actual or not three_view:
        return None
    prior_lift = three_view["integrated"]["lift"]

    deviations: list[dict[str, Any]] = []
    for cond, info in actual["by_cond"].items():
        p_lift = prior_lift.get(cond)
        if p_lift is None:
            continue
        a_lift = info["individual_lift"]
        deviations.append({
            "cond": cond,
            "label": _condition_label(cond),
            "n_actual": info["n"],
            "win_actual": info["win"],
            "actual_win_rate": info["win_rate"],
            "actual_lift": a_lift,
            "prior_lift": p_lift,
            "deviation_lift": a_lift - p_lift,
            "deviation_win_rate_pct": (info["win_rate"] - p_lift * actual["win_rate_total"]) * 100,
        })
    deviations.sort(key=lambda x: x["deviation_lift"], reverse=True)
    over = [d for d in deviations if d["deviation_lift"] > 0.10][:top_n]
    under = [d for d in deviations if d["deviation_lift"] < -0.10][-top_n:][::-1]
    return {
        "horse_id": str(horse_id),
        "actual": {
            "n_total": actual["n_total"],
            "win_total": actual["win_total"],
            "win_rate_total": actual["win_rate_total"],
            "by_cond": actual["by_cond"],
        },
        "prior_lift": prior_lift,
        "prior_source_layer": three_view["integrated"]["source_layer"],
        "deviations": deviations,
        "top_overperform": over,
        "top_underperform": under,
    }


def analyze_horse(name: str) -> dict[str, Any]:
    """馬名 → 完全な分析結果 (主流 + L1_sub + L2 + 強み/弱み)。"""
    art = _load_artifacts()
    if art is None:
        return {
            "status": "error",
            "message": (
                "アーティファクト未生成。先に "
                "`python -m src.research.pedigree.build_meta_cluster_artifacts` を実行してください。"
            ),
        }

    hid, matched, cands = find_horse_id(name)
    if hid is None:
        return {
            "status": "not_found",
            "query": name,
            "message": f"「{name}」が race テーブルに見つかりません",
        }

    sire_id, sire_name, dam_sire = get_sire_from_ped(hid)
    if sire_id is None:
        return {
            "status": "no_pedigree",
            "query": name,
            "horse_id": hid,
            "horse_name": matched,
            "message": f"5 代血統 JSON が無い (data/local/horse_pedigree_5gen/{hid[:4]}/{hid}.json)",
            "sire_name": sire_name,
            "dam_sire": dam_sire,
        }

    info = lookup_stallion(sire_id)
    if info is None:
        info = assign_unknown_stallion(sire_id)
        if info is None:
            return {
                "status": "no_cluster",
                "query": name,
                "horse_id": hid,
                "horse_name": matched,
                "sire_id": sire_id,
                "sire_name": sire_name,
                "dam_sire": dam_sire,
                "message": "父がクラスタリング対象 (>=30頭) ではなく、KNN 割当も不可 (実績不足)",
            }

    info["status"] = "ok"
    info["query"] = name
    info["horse_id"] = hid
    info["horse_name"] = matched
    info["candidates"] = cands if matched != name else []
    info["dam_sire"] = dam_sire
    info["sire_name"] = info.get("sire_name") or sire_name

    L2 = info.get("L2")
    if L2 is not None:
        info["profile"] = _profile_summary(int(L2), art)
    info["L2_meta"] = get_l2_meta(L2)

    # 役割（父・母父・母母父・父父）別 lift と blended プロファイル
    role_blend = compute_role_blended_profile(hid, art=art)
    if role_blend is not None:
        info["role_blend"] = role_blend

    # 3 ビュー（父系 / 母系 / 統合）プロファイル
    three_view = compute_three_view_profile(hid, art=art)
    if three_view is not None:
        info["three_view_profile"] = three_view

    # 個体実績 vs 血統 prior 比較
    indiv_vs = compute_individual_vs_prior(hid, art=art)
    if indiv_vs is not None:
        info["individual_vs_prior"] = indiv_vs

    # 統計的ラベル付き戦績テーブル
    rec_labeled = compute_record_with_labels(hid, art=art)
    if rec_labeled is not None:
        info["record_with_labels"] = rec_labeled

    # 特殊適性タグ (種牡馬個別 + クラスタ全体)
    tags_art = _load_special_tags()
    if tags_art is not None and sire_id:
        info["special_tags"] = tags_art["sire_tags"].get(str(sire_id), [])
        if L2 is not None:
            info["cluster_tags"] = _cluster_tags_summary(int(L2), tags_art)

    return info


def analyze_stallion(stallion_id: str) -> dict[str, Any]:
    """種牡馬 ID 自身を入力として、その種牡馬のクラスタ分類 + プロファイル + タグを返す。

    pedigree-map ツリーから種牡馬名をクリックされた際の遷移先 (/bloodline-cluster?stallion_id=...)
    で利用される。lookup_by_horse_id とは異なり「入力 ID = 解析対象種牡馬」として扱う。
    """
    art = _load_artifacts()
    if art is None:
        return {"status": "error", "message": "アーティファクト未生成"}
    sid = str(stallion_id).strip()
    info = lookup_stallion(sid) or assign_unknown_stallion(sid)
    sname = art.get("sid_to_name", {}).get(sid)
    if info is None:
        return {
            "status": "no_cluster",
            "stallion_id": sid,
            "stallion_name": sname or sid,
            "message": "クラスタ判定不可 (種牡馬未登録 or 産駒成績不足)",
        }
    info["status"] = "ok"
    info["stallion_id"] = sid
    info["stallion_name"] = info.get("sire_name") or sname or sid
    info.setdefault("sire_id", sid)
    info.setdefault("sire_name", info["stallion_name"])
    L2 = info.get("L2")
    if L2 is not None:
        info["profile"] = _profile_summary(int(L2), art)
    info["L2_meta"] = get_l2_meta(L2)

    # 種牡馬自身の role 別 lift (F=自分、MF=自分が母父役、FF=自分が父父役、MMF=自分が母母父役)
    stallion_role_lift: dict[str, Any] = {}
    for role in ROLE_BLEND_WEIGHTS_BASE:
        prof = _role_lift_for(sid, role, art)
        if prof is None:
            continue
        sp = sorted(prof["lift"].items(), key=lambda x: x[1])

        def _entry(c: str, v: float) -> dict[str, Any]:
            return {
                "cond": c,
                "label": _condition_label(c),
                "lift": float(v),
                "lift_pct": (float(v) - 1.0) * 100.0,
            }
        prof["strengths"] = [_entry(c, v) for c, v in sp[-7:][::-1]]
        prof["weaknesses"] = [_entry(c, v) for c, v in sp[:7]]
        stallion_role_lift[role] = prof
    if stallion_role_lift:
        info["role_lift_self"] = stallion_role_lift

    tags_art = _load_special_tags()
    if tags_art is not None:
        info["special_tags"] = tags_art["sire_tags"].get(sid, [])
        if L2 is not None:
            info["cluster_tags"] = _cluster_tags_summary(int(L2), tags_art)

    # full タグ (緩い閾値版) も併せて返す。pedigree-map から来る場合
    # こちらの方がカバレッジが広い。
    full_tags = get_sire_tags_full_map().get(sid, [])
    if full_tags:
        info["special_tags_full"] = full_tags
    return info


def lookup_by_horse_id(horse_id: str) -> dict[str, Any]:
    """horse_id → 完全な分析結果"""
    art = _load_artifacts()
    if art is None:
        return {"status": "error", "message": "アーティファクト未生成"}
    horse_id = str(horse_id)
    matched = art["id_to_name"].get(horse_id) or "(race テーブル外)"
    sire_id, sire_name, dam_sire = get_sire_from_ped(horse_id)
    if sire_id is None:
        return {
            "status": "no_pedigree",
            "horse_id": horse_id,
            "horse_name": matched,
            "message": f"5 代血統 JSON が無い (id={horse_id})",
        }
    info = lookup_stallion(sire_id) or assign_unknown_stallion(sire_id)
    if info is None:
        return {
            "status": "no_cluster",
            "horse_id": horse_id,
            "horse_name": matched,
            "sire_id": sire_id,
            "sire_name": sire_name,
            "dam_sire": dam_sire,
            "message": "クラスタ判定不可",
        }
    info["status"] = "ok"
    info["horse_id"] = horse_id
    info["horse_name"] = matched
    info["dam_sire"] = dam_sire
    info["sire_name"] = info.get("sire_name") or sire_name
    L2 = info.get("L2")
    if L2 is not None:
        info["profile"] = _profile_summary(int(L2), art)
    info["L2_meta"] = get_l2_meta(L2)

    role_blend = compute_role_blended_profile(horse_id, art=art)
    if role_blend is not None:
        info["role_blend"] = role_blend

    three_view = compute_three_view_profile(horse_id, art=art)
    if three_view is not None:
        info["three_view_profile"] = three_view
    indiv_vs = compute_individual_vs_prior(horse_id, art=art)
    if indiv_vs is not None:
        info["individual_vs_prior"] = indiv_vs
    rec_labeled = compute_record_with_labels(horse_id, art=art)
    if rec_labeled is not None:
        info["record_with_labels"] = rec_labeled

    tags_art = _load_special_tags()
    if tags_art is not None and sire_id:
        info["special_tags"] = tags_art["sire_tags"].get(str(sire_id), [])
        if L2 is not None:
            info["cluster_tags"] = _cluster_tags_summary(int(L2), tags_art)
    return info


def _cluster_tags_summary(L2: int, tags_art: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    """L2 クラスタの特化タグ要約 (各タグの所属種牡馬中 top20 比率 上位)。"""
    items = tags_art.get("cluster_tags", {}).get(str(L2), [])
    valid = [
        x for x in items
        if x.get("n_eligible", 0) > 0 and x.get("pct_in_top20") is not None
    ]
    valid.sort(key=lambda x: (x["pct_in_top20"], x.get("mean_lift") or 0), reverse=True)
    return valid[:top_n]


L1_FOUNDERS: dict[str, dict[str, str]] = {
    "Turn-To系":         {"id": "000a001042", "name": "Turn-to",         "year": "1951"},
    "Native Dancer系":   {"id": "000a000f89", "name": "Native Dancer",   "year": "1950"},
    "Northern Dancer系": {"id": "000a000e04", "name": "Northern Dancer", "year": "1961"},
    "Nasrullah系":       {"id": "000a000f88", "name": "Nasrullah",       "year": "1940"},
}

L1_DESCRIPTIONS: dict[str, str] = {
    "Turn-To系":
        "Hail to Reason → Roberto / Halo を経由する世界的潮流。Sunday Silence の祖父 Halo もここ。"
        "日本では Deep Impact, Heart's Cry, Just A Way 等が代表。",
    "Native Dancer系":
        "Raise a Native → Mr. Prospector で爆発的に拡大。米国ダート王道だが芝適性も高い。"
        "日本では Kingmambo → Lord Kanaloa / King Kamehameha → Ruler Ship 等。",
    "Northern Dancer系":
        "現代世界の主流。Northern Dancer → Sadler's Wells / Storm Bird / Nijinsky 等から派生。"
        "日本では Henny Hughes(Storm Cat 系) のダート系などが活躍。",
    "Nasrullah系":
        "Nasrullah → Princely Gift / Bold Ruler / Never Bend。古典系統で芝・ダート両方に影響。"
        "現在は Sinister Minister(Bold Ruler 系) などダート短中距離で目立つ。",
    "非主流":
        "4 大主流の派生に属さない種牡馬群。Monsun系, Singspiel, Wild Rush系 等。"
        "個別系統群として扱う。",
}


def get_cluster_hierarchy() -> dict[str, Any]:
    """L1 (系統) × L2 (適性) × L3 (細粒度) の階層構造を返す。

    第一階層 = 4 大主流血統 + 非主流 (全 4,773 種牡馬の系統分類が起点)。
    L2 ラベルは主要種牡馬 (n_horses 十分) のみに付与され、それ以外は L2 未分類。

    Returns:
        {
          "L1_groups": [{"name", "founder_id", "founder_name", "color",
                          "n_stallions_total", "n_stallions_with_L2",
                          "l2_distribution", "top_main_stallions",
                          "description"}, ...],
          "L2_clusters": [...],
          "matrix": [...],
          "L3_clusters": [...]
        }
    """
    art = _load_artifacts()
    if art is None:
        return {"L1_groups": [], "L2_clusters": [], "matrix": [], "L3_clusters": []}
    unified = art["unified"]
    stallions = unified[unified["entity_type"] == "stallion"].copy()
    stallions["main_group"] = stallions["main_group"].astype(str)
    # 拡張ネームマップ (海外祖先含む)
    full_name_map: dict[str, str] = {}
    full_name_path = ART_DIR / "sid_to_name_full.json"
    if full_name_path.exists():
        try:
            full_name_map = json.loads(full_name_path.read_text(encoding="utf-8"))
        except Exception:
            full_name_map = {}

    # 全種牡馬の系統分布 (sid_to_main は ~4,773 頭をカバー)
    sid_to_main = art.get("sid_to_main", {}) or {}
    total_by_main: dict[str, int] = {}
    for mg in sid_to_main.values():
        total_by_main[mg] = total_by_main.get(mg, 0) + 1

    # L1 (系統血統) — 主要 4 系統 + 非主流
    L1_COLORS = {
        "Turn-To系": "#3b82f6",
        "Native Dancer系": "#10b981",
        "Northern Dancer系": "#f59e0b",
        "Nasrullah系": "#ef4444",
        "非主流": "#9ca3af",
    }
    L1_ICONS = {
        "Turn-To系": "△",
        "Native Dancer系": "○",
        "Northern Dancer系": "◇",
        "Nasrullah系": "□",
        "非主流": "✕",
    }
    # 順序: 4 大主流 → 非主流 (種牡馬数で並べる訳ではない)
    L1_ORDER = ["Turn-To系", "Native Dancer系", "Northern Dancer系", "Nasrullah系", "非主流"]
    l1_groups: list[dict[str, Any]] = []
    for mg in L1_ORDER:
        founder = L1_FOUNDERS.get(mg)
        sub_l2 = stallions[stallions["main_group"] == mg]
        # L2 分布
        l2_dist = {}
        if len(sub_l2) > 0:
            for L2_val, cnt in sub_l2["L2"].value_counts().items():
                l2_dist[int(L2_val)] = int(cnt)
        # 主要種牡馬上位 (n_horses 多い順)
        top_main = sub_l2.sort_values("n_horses", ascending=False).head(10)[
            ["entity_id", "entity_label", "n_horses", "L2"]
        ].to_dict(orient="records")
        top_main_clean = [
            {"id": str(r["entity_id"]),
             "name": str(r["entity_label"])[:24],
             "n_horses": int(r["n_horses"]),
             "L2": int(r["L2"])}
            for r in top_main
        ]
        # 非主流 group エンティティ (Monsun 系等) の追加情報
        nonmain_subgroups: list[dict[str, Any]] = []
        if mg == "非主流":
            nm = unified[unified["entity_type"] == "group"]
            for _, r in nm.sort_values("n_horses", ascending=False).iterrows():
                nonmain_subgroups.append({
                    "id": str(r["entity_id"]),
                    "name": str(r["entity_label"]),
                    "n_horses": int(r["n_horses"]),
                })

        l1_groups.append({
            "name": mg,
            "is_nonmain": mg == "非主流",
            "founder_id": founder["id"] if founder else None,
            "founder_name": founder["name"] if founder else None,
            "founder_year": founder["year"] if founder else None,
            "color": L1_COLORS.get(mg, "#888"),
            "icon": L1_ICONS.get(mg, "●"),
            "description": L1_DESCRIPTIONS.get(mg, ""),
            "n_stallions_total": int(total_by_main.get(mg, 0)),
            "n_stallions_with_L2": int(len(sub_l2)),
            "l2_distribution": l2_dist,
            "top_main_stallions": top_main_clean,
            "nonmain_subgroups": nonmain_subgroups,
        })

    # L2 (適性領域)
    l2_clusters: list[dict[str, Any]] = []
    for L2 in sorted(art["l2_profiles"].keys()):
        sub = stallions[stallions["L2"] == L2]
        meta = get_l2_meta(int(L2)) or {}
        top = sub.sort_values("n_horses", ascending=False).head(8)[
            ["entity_id", "entity_label", "main_group"]
        ].to_dict(orient="records")
        l2_clusters.append({
            "L2": int(L2),
            "name": meta.get("name", f"L2={L2}"),
            "subtitle": meta.get("subtitle", ""),
            "color": meta.get("color", "#888"),
            "icon": meta.get("icon", "●"),
            "highlights": meta.get("highlights", []),
            "weaknesses": meta.get("weaknesses", []),
            "rationale": meta.get("rationale", ""),
            "n_stallions": int(len(sub)),
            "top_members": [
                {"id": str(r["entity_id"]), "name": str(r["entity_label"])[:24],
                 "L1": str(r["main_group"])}
                for r in top
            ],
        })

    # L1 × L2 マトリクス
    matrix: list[dict[str, Any]] = []
    for mg in [g["name"] for g in l1_groups if not g.get("is_nonmain")]:
        for L2 in sorted(art["l2_profiles"].keys()):
            cell = stallions[
                (stallions["main_group"] == mg) & (stallions["L2"] == L2)
            ]
            if len(cell) == 0:
                matrix.append({
                    "L1": mg, "L2": int(L2),
                    "n_stallions": 0, "top_members": [],
                })
                continue
            top = cell.sort_values("n_horses", ascending=False).head(5)[
                ["entity_id", "entity_label", "n_horses"]
            ]
            matrix.append({
                "L1": mg,
                "L2": int(L2),
                "n_stallions": int(len(cell)),
                "top_members": [
                    {"id": str(r["entity_id"]),
                     "name": str(r["entity_label"])[:24],
                     "n": int(r["n_horses"])}
                    for _, r in top.iterrows()
                ],
            })

    # Super-groups (L2_fine の親グルーピング) + 2D 座標 + 類似度
    super_groups: list[dict[str, Any]] = []
    sg_meta = art.get("super_groups_meta", {}) or {}
    sg_artifact = art.get("l2_super_groups", {}) or {}
    l2_to_super: dict[int, int] = {
        int(k): int(v) for k, v in sg_artifact.get("L2_to_super", {}).items()
    }
    if sg_meta:
        for sg_id, info in sorted(sg_meta.items(), key=lambda x: int(x[0])):
            members = [int(m) for m in info.get("member_L2_ids", [])]
            super_groups.append({
                "super_id": int(sg_id),
                "name": info.get("name"),
                "subtitle": info.get("subtitle"),
                "color": info.get("color"),
                "icon": info.get("icon"),
                "n_L2": int(info.get("n_L2", len(members))),
                "n_stallions": int(info.get("n_stallions", 0)),
                "member_L2_ids": members,
                "member_L2_names": info.get("member_L2_names", []),
                "top_stallions": info.get("top_stallions", []),
                "top_conditions": info.get("top_conditions", []),
            })

    # L2 ⇄ super_group の紐付けを L2_clusters に追加
    for c in l2_clusters:
        c["super_group"] = l2_to_super.get(c["L2"], 0)

    # 2D 座標 (PCA + UMAP)
    pos_artifact = art.get("l2_positions_2d", {}) or {}
    positions_2d: dict[str, Any] = {}
    for proj in ("pca", "umap"):
        proj_data = pos_artifact.get(proj, {})
        if not proj_data or "error" in proj_data:
            continue
        positions_2d[proj] = {
            "centroids": [
                {"L2": int(k), "x": float(v[0]), "y": float(v[1])}
                for k, v in proj_data.get("centroids", {}).items()
            ],
            "stallions": [
                {"id": str(k), "x": float(v[0]), "y": float(v[1])}
                for k, v in proj_data.get("stallions", {}).items()
            ],
        }
        if "explained_variance_ratio" in proj_data:
            positions_2d[proj]["explained_variance_ratio"] = proj_data["explained_variance_ratio"]

    # 主流外祖先の 2D 座標 (ancestor_positions_2d.json + ancestor_to_l2.json から)
    anc_pos = art.get("ancestor_positions_2d", {}) or {}
    anc_l2_map = art.get("ancestor_to_l2", {}) or {}
    # 主流種牡馬の id 集合 (positions_2d に既にある = 重複避け)
    main_ids: set[str] = set()
    for proj in ("pca", "umap"):
        for s in positions_2d.get(proj, {}).get("stallions", []):
            main_ids.add(s["id"])
    for proj in ("pca", "umap"):
        proj_data = anc_pos.get(proj)
        if not proj_data:
            continue
        positions = proj_data.get("positions", {})
        if not positions:
            continue
        anc_points = []
        for sid, xy in positions.items():
            if sid in main_ids:
                continue
            info = anc_l2_map.get(sid, {})
            anc_points.append({
                "id": sid,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "L2": int(info.get("L2", -1)),
                "name": str(info.get("name", "")),
                "main_group": str(info.get("main_group", "")),
                "method": str(info.get("method", "")),
            })
        positions_2d.setdefault(proj, {})["ancestors"] = anc_points

    # 類似度
    sim_artifact = art.get("l2_similarity", {}) or {}
    similarity: dict[str, Any] = {}
    if sim_artifact:
        similarity = {
            "cos_sim_matrix": sim_artifact.get("cos_sim_matrix", []),
            "nearest_top3": sim_artifact.get("nearest_top3", {}),
            "farthest_top3": sim_artifact.get("farthest_top3", {}),
        }

    nonmain = unified[unified["entity_type"] == "group"]
    return {
        "L1_groups": l1_groups,
        "L2_clusters": l2_clusters,
        "matrix": matrix,
        "super_groups": super_groups,
        "positions_2d": positions_2d,
        "similarity": similarity,
        "L2_to_super": {str(k): int(v) for k, v in l2_to_super.items()},
        "totals": {
            "n_stallions_total": int(sum(total_by_main.values())),
            "n_stallions_with_L2": int(len(stallions)),
            "n_nonmain_subgroups": int(len(nonmain)),
            "n_L1": 4,
            "n_L2_fine": int(len(l2_clusters)),
            "n_super_groups": int(len(super_groups)),
        },
    }


def list_clusters() -> list[dict[str, Any]]:
    """全 L2 メタクラスタの構成 + プロファイル + 特化タグ。"""
    art = _load_artifacts()
    if art is None:
        return []
    unified = art["unified"]
    tags_art = _load_special_tags()
    out: list[dict[str, Any]] = []
    for L2 in sorted(art["l2_profiles"].keys()):
        sub = unified[unified["L2"] == L2].copy()
        sub_sorted = sub.sort_values("n_horses", ascending=False)
        prof = _profile_summary(int(L2), art)
        by_main: dict[str, list[dict[str, Any]]] = {}
        for _, r in sub_sorted.iterrows():
            mg = str(r["main_group"])
            by_main.setdefault(mg, []).append({
                "entity_id": str(r["entity_id"]),
                "entity_label": str(r["entity_label"]),
                "L1_sub": int(r["L1_sub"]),
                "n_horses": int(r["n_horses"]),
            })
        item = {
            "L2": int(L2),
            "L2_meta": get_l2_meta(int(L2)),
            "n_entities": int(len(sub)),
            "n_horses_total": int(sub["n_horses"].sum()),
            "main_group_counts": sub.groupby("main_group").size().to_dict(),
            "by_main_group": by_main,
            "strengths": prof["strengths"] if prof else [],
            "weaknesses": prof["weaknesses"] if prof else [],
        }
        if tags_art is not None:
            item["special_tags"] = _cluster_tags_summary(int(L2), tags_art, top_n=5)
        out.append(item)
    return out


# ── タグ別カラー / カテゴリ ─────────────────────────
TAG_CATEGORY_COLORS: dict[str, str] = {
    "脚質":   "#ff6b6b",  # 赤系
    "馬場":   "#4ecdc4",  # 青緑
    "ペース": "#ffe66d",  # 黄
    "距離":   "#a78bfa",  # 紫
    "コース": "#fb923c",  # オレンジ
    "枠順":   "#60a5fa",  # 青
    "上がり": "#34d399",  # 緑
    "ローテ": "#f472b6",  # ピンク
    "季節":   "#fbbf24",  # 琥珀
    "年齢":   "#c084fc",  # 薄紫
}


def get_sire_tags_map(min_lift: float = 1.0) -> dict[str, list[dict[str, Any]]]:
    """全種牡馬の Top20% タグマップ ({stallion_id: [tag, ...]})。

    pedigree-map のような '全種牡馬' 系ビュー向けに、軽量タグデータを返す。
    各タグには category と category_color が付与される。
    """
    tags_art = _load_special_tags()
    if tags_art is None:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, tags in tags_art.get("sire_tags", {}).items():
        items = []
        for t in tags:
            if t.get("lift", 0) < min_lift:
                continue
            items.append({
                **t,
                "category_color": TAG_CATEGORY_COLORS.get(t.get("category"), "#888"),
            })
        if items:
            out[str(sid)] = items
    return out


# ── ツリー用 (全種牡馬版) タグマップ - lazily loaded ──────────────────────
def _load_full_stallion_tags() -> Optional[dict[str, list[dict[str, Any]]]]:
    """stallion_tags_full.parquet を lazily ロードし {stallion_id: [tag, ...]} を返す。

    is_top20 のみを残し、各タグに category_color を付与済の軽量フォーマット。
    """
    art = _load_artifacts()
    if art is None:
        # アーティファクトが無い場合でも、parquet 単体だけは読めるかもしれない
        from pathlib import Path
        full_path = ART_DIR / "stallion_tags_full.parquet"
        if not full_path.exists():
            return None
        art = {}
    if "full_tags_loaded" in art:
        return art.get("sire_tags_full", {})
    full_path = ART_DIR / "stallion_tags_full.parquet"
    if not full_path.exists():
        logger.warning("stallion_tags_full.parquet 未生成: %s", full_path)
        art["full_tags_loaded"] = True
        art["sire_tags_full"] = {}
        return {}
    def _sf(v) -> float | None:
        try:
            return None if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return None

    def _si(v) -> int | None:
        try:
            return None if pd.isna(v) else int(v)
        except (TypeError, ValueError):
            return None

    with _lock:
        if "full_tags_loaded" in art:
            return art.get("sire_tags_full", {})
        tdf = pd.read_parquet(full_path)
        tdf["stallion_id"] = tdf["stallion_id"].astype(str)
        out: dict[str, list[dict[str, Any]]] = {}
        for sid, g in tdf[tdf["is_top20"]].sort_values("combo_score", ascending=False).groupby("stallion_id"):
            out[sid] = [{
                "tag_id": r["tag_id"],
                "tag_label": r["tag_label"],
                "category": r["category"],
                "rate_eb": _sf(r.get("rate_eb")),
                "rate_raw": _sf(r.get("rate_raw")),
                "lift": _sf(r.get("lift")),
                "n_records": _si(r.get("n_records")),
                "wins": _si(r.get("wins")),
                "baseline_eb": _sf(r.get("baseline_eb")),
                "cond_global": _sf(r.get("cond_global")),
                "share": _sf(r.get("share")),
                "sire_n_total": _si(r.get("sire_n_total")),
                "is_top20": bool(r.get("is_top20", False)),
                "is_strong_abs": bool(r.get("is_strong_abs", False)),
                "is_relative_strong": bool(r.get("is_relative_strong", False)),
                "is_specialist": bool(r.get("is_specialist", False)) if "is_specialist" in r and pd.notna(r["is_specialist"]) else None,
                "category_color": TAG_CATEGORY_COLORS.get(r["category"], "#888"),
            } for _, r in g.iterrows()]
        art["sire_tags_full"] = out
        art["full_tags_loaded"] = True
        logger.info(
            "full tags ready: %d 種牡馬 (合計 %d タグ)",
            len(out), int(tdf["is_top20"].sum()),
        )
    return out


def get_sire_tags_full_map() -> dict[str, list[dict[str, Any]]]:
    """ツリー用: 全種牡馬 (緩めの閾値) のタグマップ。"""
    res = _load_full_stallion_tags()
    return res if res is not None else {}


def get_horse_tags_map(min_lift: float = 1.0) -> dict[str, list[dict[str, Any]]]:
    """全 horse_id → 種牡馬タグ ({horse_id: [tag, ...]}) を返す。

    pedigree-map のノードは horse_id ベースなので、stallion_id ではなく
    horse_id (ノード自身) → そのノードが持つ "種牡馬としてのタグ" を返す。
    """
    sire_map = get_sire_tags_map(min_lift=min_lift)
    if not sire_map:
        return {}
    art = _load_artifacts()
    if art is None:
        return sire_map
    h2s = art.get("h2s")
    if h2s is None:
        h2s_path = ART_DIR / "horse_to_sire.parquet"
        if h2s_path.exists():
            h2s = pd.read_parquet(h2s_path)
            art["h2s"] = h2s
    out: dict[str, list[dict[str, Any]]] = {}
    if h2s is not None:
        sids_with_tags = set(sire_map.keys())
        for sid in sids_with_tags:
            out[sid] = sire_map[sid]
    else:
        out = sire_map
    return out


def list_tag_definitions() -> list[dict[str, str]]:
    """利用可能な全タグ定義 (id, label, category, color) を返す。"""
    tags_art = _load_special_tags()
    if tags_art is None:
        return []
    items: list[dict[str, str]] = []
    for tag_id, meta in tags_art.get("tag_meta", {}).items():
        items.append({
            "tag_id": tag_id,
            "label": meta["label"],
            "category": meta["category"],
            "category_color": TAG_CATEGORY_COLORS.get(meta["category"], "#888"),
        })
    return items


def list_special_tags(
    tag_id: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 30,
) -> dict[str, Any]:
    """特殊適性タグの一覧 + 各タグでの上位種牡馬。

    引数:
        tag_id   : 指定したタグのみ返す (省略時は全タグ)
        category : 指定したカテゴリ (脚質/馬場/ペース/距離/コース) のみ
        top_n    : 各タグの上位 N 件 (combo_score 降順)
    """
    tags_art = _load_special_tags()
    if tags_art is None:
        return {"status": "error", "message": "stallion_tags.parquet 未生成"}
    tdf = tags_art["tags_df"]
    art = _load_artifacts()
    sid_to_main = art["sid_to_main"] if art else {}

    if tag_id:
        tdf = tdf[tdf["tag_id"] == tag_id]
    if category:
        tdf = tdf[tdf["category"] == category]

    def _sf(v) -> float | None:
        try:
            return None if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return None

    def _si(v) -> int | None:
        try:
            return None if pd.isna(v) else int(v)
        except (TypeError, ValueError):
            return None

    out: dict[str, dict[str, Any]] = {}
    for tid, g in tdf.groupby("tag_id"):
        first = g.iloc[0]
        top_rows = g[g["is_top20"]].sort_values("combo_score", ascending=False).head(top_n)
        items = [{
            "stallion_id": str(r["stallion_id"]),
            "stallion_name": r["stallion_name"],
            "main_group": sid_to_main.get(str(r["stallion_id"])),
            "rate_eb": _sf(r.get("rate_eb")),
            "rate_raw": _sf(r.get("rate_raw")),
            "lift": _sf(r.get("lift")),
            "n_records": _si(r.get("n_records")),
            "wins": _si(r.get("wins")),
            "pct_rank": _sf(r.get("pct_rank")),
            # ポップアップ詳細表示用の追加統計
            "baseline_eb": _sf(r.get("baseline_eb")),
            "cond_global": _sf(r.get("cond_global")),
            "share": _sf(r.get("share")),
            "sire_n_total": _si(r.get("sire_n_total")),
            "is_top20": bool(r.get("is_top20", False)),
            "is_strong_abs": bool(r.get("is_strong_abs", False)),
            "is_relative_strong": bool(r.get("is_relative_strong", False)),
            "is_specialist": bool(r.get("is_specialist", False)) if "is_specialist" in r and pd.notna(r["is_specialist"]) else None,
        } for _, r in top_rows.iterrows()]
        out[str(tid)] = {
            "tag_id": str(tid),
            "tag_label": str(first["tag_label"]),
            "category": str(first["category"]),
            "n_eligible": int(len(g)),
            "n_top20": int(g["is_top20"].sum()),
            "top_stallions": items,
        }
    return {"status": "ok", "tags": out, "total_tags": len(out)}


def get_meta() -> Optional[dict[str, Any]]:
    art = _load_artifacts()
    return art["meta"] if art else None


def list_horses(prefix: str = "", limit: int = 30) -> list[dict[str, str]]:
    """馬名インデックスを前方一致 / 部分一致で列挙 (オートサジェスト用)。"""
    art = _load_artifacts()
    if art is None:
        return []
    prefix = (prefix or "").strip()
    if not prefix:
        return []
    n2i = art["name_to_id"]
    starts = [n for n in n2i if n.startswith(prefix)]
    contains = [n for n in n2i if prefix in n and not n.startswith(prefix)]
    cands = (starts + contains)[:limit]
    return [{"name": n, "horse_id": n2i[n]} for n in cands]


# ══════════════════════════════════════════════════════
#  統計集計 (期間 × 舞台 × 種牡馬/クラスタ)
# ══════════════════════════════════════════════════════


def _load_special_tags() -> Optional[dict[str, Any]]:
    """特殊適性タグを遅延ロード。
    cache 内に:
        tags_df : stallion_tags.parquet
        sire_tags: { stallion_id: [tag dict, ...] }  (is_top20 のみ)
        cluster_tags: { L2: [tag summary, ...] }
        tag_meta : { tag_id: {label, category} }
    """
    art = _load_artifacts()
    if art is None:
        return None
    if "tags_loaded" in art:
        return art
    tags_path = ART_DIR / "stallion_tags.parquet"
    cluster_tags_path = ART_DIR / "cluster_tags.json"
    if not tags_path.exists():
        logger.warning("stallion_tags.parquet 未生成: %s", tags_path)
        return None
    with _lock:
        if "tags_loaded" in art:
            return art
        tdf = pd.read_parquet(tags_path)
        tdf["stallion_id"] = tdf["stallion_id"].astype(str)
        # 各種牡馬の Top20% タグだけ集約 (UI のポップアップで詳細統計を表示できるよう全項目を返す)
        def _safe_float(v) -> float | None:
            try:
                if pd.isna(v):
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None

        def _safe_int(v) -> int | None:
            try:
                if pd.isna(v):
                    return None
                return int(v)
            except (TypeError, ValueError):
                return None

        sire_tags: dict[str, list[dict[str, Any]]] = {}
        for sid, g in tdf[tdf["is_top20"]].sort_values("combo_score", ascending=False).groupby("stallion_id"):
            sire_tags[sid] = [{
                "tag_id": r["tag_id"],
                "tag_label": r["tag_label"],
                "category": r["category"],
                "n_records": _safe_int(r.get("n_records")),
                "wins": _safe_int(r.get("wins")),
                "rate_raw": _safe_float(r.get("rate_raw")),
                "rate_eb": _safe_float(r.get("rate_eb")),
                "baseline_eb": _safe_float(r.get("baseline_eb")),
                "cond_global": _safe_float(r.get("cond_global")),
                "lift": _safe_float(r.get("lift")),
                "share": _safe_float(r.get("share")),
                "sire_n_total": _safe_int(r.get("sire_n_total")),
                "combo_score": _safe_float(r.get("combo_score")),
                "pct_rank": _safe_float(r.get("pct_rank")),
                "is_top20": bool(r.get("is_top20", False)),
                "is_strong_abs": bool(r.get("is_strong_abs", False)),
                "is_relative_strong": bool(r.get("is_relative_strong", False)),
                "is_specialist": bool(r.get("is_specialist", False)) if "is_specialist" in r and pd.notna(r["is_specialist"]) else None,
            } for _, r in g.iterrows()]
        cluster_tags = {}
        if cluster_tags_path.exists():
            cluster_tags = json.loads(cluster_tags_path.read_text())
        # tag_id → meta
        tag_meta = {tag_id: {"label": label, "category": category}
                    for tag_id, label, category, _ in [
                        (r["tag_id"], r["tag_label"], r["category"], None)
                        for _, r in tdf.drop_duplicates("tag_id").iterrows()
                    ]}
        art["tags_df"] = tdf
        art["sire_tags"] = sire_tags
        art["cluster_tags"] = cluster_tags
        art["tag_meta"] = tag_meta
        art["tags_loaded"] = True
        logger.info(
            "tags ready: %d 種牡馬 がタグ付き (中: top20 %d 件)",
            len(sire_tags), int(tdf["is_top20"].sum()),
        )
    return art


def _load_stats_assets() -> Optional[dict[str, Any]]:
    """statistics 用の追加アーティファクト (race_records / horse_to_sire) を遅延ロード。"""
    art = _load_artifacts()
    if art is None:
        return None
    if "stats_loaded" in art:
        return art
    rec_path = ART_DIR / "race_records.parquet"
    h2s_path = ART_DIR / "horse_to_sire.parquet"
    if not rec_path.exists() or not h2s_path.exists():
        logger.warning("stats アーティファクト未生成: %s / %s", rec_path, h2s_path)
        return None
    with _lock:
        if "stats_loaded" in art:
            return art
        rec = pd.read_parquet(rec_path)
        rec["race_id"] = rec["race_id"].astype(str)
        rec["horse_id"] = rec["horse_id"].astype(str)
        if "date" in rec.columns:
            rec["date"] = pd.to_datetime(rec["date"], errors="coerce")
        h2s = pd.read_parquet(h2s_path)
        h2s["horse_id"] = h2s["horse_id"].astype(str)
        h2s["stallion_id"] = h2s["stallion_id"].astype(str)
        sire_of: dict[str, str] = dict(zip(h2s["horse_id"], h2s["stallion_id"]))
        rec["stallion_id"] = rec["horse_id"].map(sire_of)
        art["records"] = rec
        art["sire_of_horse"] = sire_of
        art["stats_loaded"] = True
        logger.info(
            "stats records ready: %d rows (odds %.1f%% / 複勝払戻 %.1f%%)",
            len(rec),
            rec["odds"].notna().mean() * 100 if "odds" in rec.columns else 0.0,
            rec["place_payout_yen"].notna().mean() * 100 if "place_payout_yen" in rec.columns else 0.0,
        )
        return art


# JRA 公式 1 着本賞金の概算 (万円)。出走馬一覧の並び順 (獲得賞金順) 用。
# G1 はレースにより大きく異なるが、概算用に平均的な値を採用。
_PRIZE_BASE_BY_GRADE: dict[str, float] = {
    "G1":  12000.0,
    "G2":   6500.0,
    "G3":   4300.0,
    "L":    3500.0,
    "OP":   2700.0,
    "3勝":  1500.0,
    "2勝":  1060.0,
    "1勝":   590.0,
    "未勝利": 530.0,
    "新馬":  530.0,
}
# 着順別本賞金の比率 (5着まで)
_PRIZE_RATIO_BY_FINISH: dict[int, float] = {
    1: 1.00, 2: 0.40, 3: 0.25, 4: 0.15, 5: 0.10,
}


def _estimate_prize_per_record(grade: Any, finish: Any) -> float:
    """1 出走あたりの推定本賞金 (万円, 1着換算ベース)。"""
    try:
        fp = int(finish)
    except (TypeError, ValueError):
        return 0.0
    if fp < 1 or fp > 5:
        return 0.0
    base = _PRIZE_BASE_BY_GRADE.get(str(grade) if grade is not None else "", 0.0)
    if base <= 0:
        return 0.0
    return base * _PRIZE_RATIO_BY_FINISH.get(fp, 0.0)


def _load_horse_prize_map() -> Optional[dict[str, Any]]:
    """horse_id -> 累積推定賞金 (万円) を計算してキャッシュ。"""
    art = _load_stats_assets()
    if art is None:
        return None
    if "horse_prize_loaded" in art:
        return art
    with _lock:
        if "horse_prize_loaded" in art:
            return art
        rec: pd.DataFrame = art["records"]
        sub = rec[rec["finish_position"].notna() & (rec["finish_position"] > 0)]
        sub = sub[sub["finish_position"].astype(int) <= 5].copy()
        sub["_prize_man"] = [
            _estimate_prize_per_record(g, f)
            for g, f in zip(sub["grade"].astype(str), sub["finish_position"])
        ]
        prize_map = sub.groupby("horse_id")["_prize_man"].sum().to_dict()
        # 名前テーブル
        # race_records には horse_name が無いので、別途 horse 名のマスタを構築
        # (NULL の場合は horse_id をそのまま表示)
        name_map: dict[str, str] = {}
        try:
            from glob import glob
            for f in sorted(glob(str(ROOT / "data/page_reference/tables/*/race_result_flat.parquet"))):
                df = pd.read_parquet(f, columns=["horse_id", "horse_name"])
                df = df.dropna(subset=["horse_id", "horse_name"])
                for hid, nm in zip(df["horse_id"].astype(str), df["horse_name"].astype(str)):
                    if hid and nm and hid not in name_map:
                        name_map[hid] = nm
        except Exception as e:
            logger.warning("horse_name マスタ構築失敗: %s", e)
        # 出走数 / 1着回数 / 平均着順 などの簡易メタ
        sub["_win"] = (sub["finish_position"].astype(int) == 1).astype(int)
        agg = sub.groupby("horse_id").agg(
            n_records=("finish_position", "count"),
            n_win=("_win", "sum"),
        ).to_dict("index")
        # 全出走数 (5着以下含む)
        rec_all = rec[rec["finish_position"].notna() & (rec["finish_position"] > 0)]
        n_all_map = rec_all.groupby("horse_id").size().to_dict()
        art["horse_prize_man"] = prize_map
        art["horse_name_map"] = name_map
        art["horse_basic_meta"] = agg
        art["horse_n_all_map"] = n_all_map
        art["horse_prize_loaded"] = True
        logger.info(
            "horse_prize map ready: %d 馬 / 名前マスタ %d 件",
            len(prize_map), len(name_map),
        )
        return art


def _load_pedigree_10gen() -> Optional[dict[str, Any]]:
    """直近10世代の祖先 inverted index を遅延ロード。

    返却辞書:
        ``ancestor_to_father_horses`` / ``ancestor_to_mother_horses``: ancestor_id -> set[horse_id]
    """
    art = _load_artifacts()
    if art is None:
        return None
    if "ped_10gen_loaded" in art:
        return art
    if "ped_10gen_not_found" in art:
        return None
    inv_path = PED_10GEN_DIR / "ancestor_to_horses.parquet"
    if not inv_path.exists():
        logger.warning(
            "10gen ancestor index 未生成: %s\n"
            "  -> python -m src.research.pedigree.build_pedigree_10gen_index を実行してください",
            inv_path,
        )
        art["ped_10gen_not_found"] = True
        return None
    with _lock:
        if "ped_10gen_loaded" in art:
            return art
        try:
            inv = pd.read_parquet(inv_path)
        except Exception as e:
            logger.warning("10gen index 読み込み失敗: %s", e)
            return None
        ancestor_to_father: dict[str, set[str]] = {}
        ancestor_to_mother: dict[str, set[str]] = {}
        for row in inv.itertuples(index=False):
            anc = str(row.ancestor_id)
            side = str(row.side)
            raw = row.horse_ids
            if raw is None:
                horses: set[str] = set()
            else:
                horses = {str(h) for h in raw}
            if side == "father":
                ancestor_to_father[anc] = horses
            elif side == "mother":
                ancestor_to_mother[anc] = horses
        art["ancestor_to_father_horses"] = ancestor_to_father
        art["ancestor_to_mother_horses"] = ancestor_to_mother

        # ── 祖先 horse_id → 馬名 のマップを (あれば) ロード ──
        # 母系領域 (10gen) や bms 表示で「種牡馬名が horse_id のまま」になる問題を解消。
        # build_ancestor_name_map.py で 5gen / 10gen JSON 全件から事前抽出した parquet を使う。
        anc_name_path = PED_10GEN_DIR / "ancestor_id_to_name.parquet"
        anc_name_map: dict[str, str] = {}
        if anc_name_path.exists():
            try:
                df_n = pd.read_parquet(anc_name_path)
                for hid, nm in zip(df_n["horse_id"].astype(str), df_n["name"].astype(str)):
                    if hid and nm:
                        anc_name_map[hid] = nm
            except Exception as e:
                logger.warning("ancestor_id_to_name 読み込み失敗: %s", e)
        else:
            logger.warning(
                "ancestor_id_to_name.parquet 未生成: %s\n"
                "  -> python -m src.research.pedigree.build_ancestor_name_map で生成可",
                anc_name_path,
            )
        art["ancestor_id_to_name"] = anc_name_map

        art["ped_10gen_loaded"] = True
        logger.info(
            "10gen ancestor index ready: %d (father) / %d (mother) / %d ancestor names",
            len(ancestor_to_father), len(ancestor_to_mother), len(anc_name_map),
        )
        return art


def _resolve_target_stallions(
    target_kind: str,
    target_id: Optional[str] = None,
    main_group: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    """target → (stallion_id_list, label_dict)。
    target_kind:
        'sire'    : target_id = stallion_id (1 頭)
        'L2'      : target_id = L2 番号 (str/int)
        'L1_sub'  : target_id = L1_sub 番号 + main_group 必須
        'main'    : target_id = main_group_name (主流全体)
        'all'     : 全エンティティ
    """
    art = _load_artifacts()
    if art is None:
        return [], {}
    unified = art["unified"]
    sid_to_name = art["sid_to_name"]
    nm_root_of = art["root_of"]  # stallion_id -> group_id (root)

    def _expand_entity(entity_id: str, mg: str) -> list[str]:
        """unified のエンティティ → 構成 stallion_id のリスト
        - 4 大主流: entity_id がそのまま stallion_id (cat=1, gen=1)
        - 非主流: entity_id = group_id, root_of の逆引きで全 stallion_id を集める"""
        if mg == "非主流":
            return [sid for sid, root in nm_root_of.items() if str(root) == str(entity_id)]
        return [str(entity_id)]

    target_kind = (target_kind or "sire").lower()
    sids: list[str] = []
    if target_kind == "sire" and target_id:
        sids = [str(target_id)]
    elif target_kind == "l2" and target_id is not None:
        try:
            l2 = int(target_id)
        except ValueError:
            return [], {}
        sub = unified[unified["L2"] == l2]
        for _, r in sub.iterrows():
            sids.extend(_expand_entity(str(r["entity_id"]), str(r["main_group"])))
    elif target_kind == "l1_sub" and target_id is not None and main_group:
        try:
            l1 = int(target_id)
        except ValueError:
            return [], {}
        sub = unified[(unified["L1_sub"] == l1) & (unified["main_group"] == main_group)]
        for _, r in sub.iterrows():
            sids.extend(_expand_entity(str(r["entity_id"]), str(r["main_group"])))
    elif target_kind == "main" and target_id:
        sub = unified[unified["main_group"] == str(target_id)]
        for _, r in sub.iterrows():
            sids.extend(_expand_entity(str(r["entity_id"]), str(r["main_group"])))
    elif target_kind == "all":
        for _, r in unified.iterrows():
            sids.extend(_expand_entity(str(r["entity_id"]), str(r["main_group"])))

    sids = sorted(set(s for s in sids if s))
    label = {s: sid_to_name.get(s, s) for s in sids}
    return sids, label


def _build_groupby_fast(
    sub: pd.DataFrame,
    cols: list[str],
    label_fn,
    min_n: int = 30,
) -> list[dict[str, Any]]:
    """_build_groupby の vectorized 高速版。
    pandas groupby().agg() を1回呼ぶだけで全集計を完了する。
    641グループ × _compute_aggregate() ループを排除し ~10x 高速化。
    """
    if sub.empty:
        return []

    s = sub.copy()
    fp = s["finish_position"].astype("Float64")
    s["_win"] = (fp == 1).astype(np.int32)
    s["_p2"]  = (fp <= 2).astype(np.int32)
    s["_p3"]  = (fp <= 3).astype(np.int32)
    s["_2nd"] = (fp == 2).astype(np.int32)
    s["_3rd"] = (fp == 3).astype(np.int32)
    s["_45"]  = ((fp >= 4) & (fp <= 5)).astype(np.int32)
    s["_6p"]  = (fp >= 6).astype(np.int32)

    has_odds = "odds" in s.columns
    if has_odds:
        s["_wp"] = (s["odds"].astype("Float64") * 100 * s["_win"]).fillna(0)

    has_place = "field_size" in s.columns and "place_payout_yen" in s.columns
    if has_place:
        fs = s["field_size"].astype("Float64")
        eligible = (
            (fs >= 8) |
            ((fs >= 5) & (fs <= 7) & (s["_p2"] == 1)) |
            ((fs < 5) & (s["_win"] == 1))
        ).fillna(False)
        s["_elig"] = eligible.astype(np.int32)
        s["_pp"]   = s["place_payout_yen"].where(eligible, 0).fillna(0)

    agg_spec: dict[str, Any] = {
        "n_records":   ("finish_position", "count"),
        "n_win":       ("_win", "sum"),
        "n_p2":        ("_p2",  "sum"),
        "n_p3":        ("_p3",  "sum"),
        "n_2nd":       ("_2nd", "sum"),
        "n_3rd":       ("_3rd", "sum"),
        "n_4_5":       ("_45",  "sum"),
        "n_6plus":     ("_6p",  "sum"),
        "avg_finish":  ("finish_position", "mean"),
        "n_horses":    ("horse_id",  "nunique"),
        "n_races":     ("race_id",   "nunique"),
    }
    if has_odds:
        agg_spec["win_payout"] = ("_wp", "sum")
    if has_place:
        agg_spec["place_bets"]    = ("_elig", "sum")
        agg_spec["place_payout"]  = ("_pp",   "sum")

    df_agg = s.groupby(cols, observed=True).agg(**agg_spec).reset_index()
    df_agg = df_agg[df_agg["n_records"] >= min_n]

    rows: list[dict[str, Any]] = []
    for row in df_agg.itertuples(index=False):
        n = int(row.n_records)
        n_win = int(row.n_win)
        n_p2  = int(row.n_p2)
        n_p3  = int(row.n_p3)

        win_roi: float | None = None
        if has_odds and n > 0:
            win_roi = float(row.win_payout) / (n * 100.0) * 100.0

        place_roi: float | None = None
        pb = int(row.place_bets) if has_place else 0
        if has_place and pb > 0:
            place_roi = float(row.place_payout) / (pb * 100.0) * 100.0

        keys = tuple(getattr(row, c) for c in cols)
        entry: dict[str, Any] = {
            "key":    " × ".join(str(k) for k in keys),
            "label":  label_fn(keys),
            "n_records":  n,
            "n_horses":   int(row.n_horses),
            "n_races":    int(row.n_races),
            "n_win":      n_win,
            "n_place2":   n_p2,
            "n_place3":   n_p3,
            "n_1st":      n_win,
            "n_2nd":      int(row.n_2nd),
            "n_3rd":      int(row.n_3rd),
            "n_4_5":      int(row.n_4_5),
            "n_6plus":    int(row.n_6plus),
            "finish_dist_str": f"{n_win}-{int(row.n_2nd)}-{int(row.n_3rd)}-{int(row.n_4_5)}-{int(row.n_6plus)}",
            "win_rate":    n_win / n,
            "place2_rate": n_p2  / n,
            "place3_rate": n_p3  / n,
            "avg_finish":  float(row.avg_finish) if not pd.isna(row.avg_finish) else None,
            "avg_popularity": None,
            "win_roi":   win_roi,
            "place_roi": place_roi,
            "place_bets":  pb,
            "place_coverage": None,
        }
        for c, k in zip(cols, keys):
            entry[c] = str(k) if k is not None else ""
        rows.append(entry)

    return rows


def _compute_aggregate(df: pd.DataFrame) -> dict[str, Any]:
    """フィルタ済み出走 DataFrame → 統計指標 dict

    回収率は 100 円賭け基準。
    - 単勝: 全出走を分母に、1着のみオッズ × 100 を払戻として加算。
    - 複勝: 「複勝対象となる出走」を分母に。JRA 規定で 5-7 頭立ては 3 着が対象外。
            8 頭以上 → 1-3 着が対象 / 5-7 頭 → 1-2 着が対象 / 4 頭以下 → 単勝のみで複勝なし。
    """
    n = int(len(df))
    if n == 0:
        return {
            "n_records": 0, "n_horses": 0, "n_races": 0,
            "win_rate": None, "place2_rate": None, "place3_rate": None,
            "avg_finish": None, "avg_popularity": None,
            "win_roi": None, "place_roi": None,
            "n_win": 0, "n_place2": 0, "n_place3": 0,
            # 着順分布 (1着 / 2着 / 3着 / 4-5着 / 6着以下)
            "n_1st": 0, "n_2nd": 0, "n_3rd": 0, "n_4_5": 0, "n_6plus": 0,
            "finish_dist_str": "0-0-0-0-0",
            "place_bets": 0, "place_coverage": None,
        }

    fp = df["finish_position"].astype("Float64")
    pop = df["popularity"].astype("Float64") if "popularity" in df.columns else None
    n_win = int((fp == 1).sum())
    n_p2 = int((fp <= 2).sum())
    n_p3 = int((fp <= 3).sum())
    # 各着の単独件数 (4-5着 / 6着以下も含む)
    n_2nd = int((fp == 2).sum())
    n_3rd = int((fp == 3).sum())
    n_4_5 = int(((fp >= 4) & (fp <= 5)).sum())
    n_6plus = int((fp >= 6).sum())

    odds = df["odds"].astype("Float64") if "odds" in df.columns else None
    bet_yen_win = n * 100.0
    if odds is not None:
        win_payout_yen = float((odds.where(fp == 1, 0) * 100).fillna(0).sum())
        win_roi = win_payout_yen / bet_yen_win * 100.0
    else:
        win_roi = None

    place_roi = None
    place_bets = 0
    place_coverage = None
    if "place_payout_yen" in df.columns and "field_size" in df.columns:
        fs = df["field_size"].astype("Float64")
        eligible = (
            (fs >= 8) |  # 3 着以内すべて対象
            ((fs >= 5) & (fs <= 7) & (fp <= 2)) |  # 1-2 着のみ対象
            ((fs < 5) & (fp == 1))  # ほぼ無いが 1 着のみ
        )
        eligible = eligible.fillna(False)
        bets = int(eligible.sum())
        place_bets = bets
        if bets > 0:
            payout = float(df.loc[eligible.astype(bool), "place_payout_yen"].fillna(0).sum())
            place_roi = payout / (bets * 100.0) * 100.0
            # in_money かつ payout 取得済みの率 (データ品質指標)
            in_money = eligible & (fp <= 3)
            in_money_n = int(in_money.sum())
            if in_money_n > 0:
                got = int(df.loc[in_money.astype(bool), "place_payout_yen"].notna().sum())
                place_coverage = got / in_money_n

    return {
        "n_records": n,
        "n_horses": int(df["horse_id"].nunique()),
        "n_races": int(df["race_id"].nunique()),
        "n_win": n_win, "n_place2": n_p2, "n_place3": n_p3,
        # 着順分布: 1着 / 2着 / 3着 / 4-5着 / 6着以下
        "n_1st": n_win, "n_2nd": n_2nd, "n_3rd": n_3rd,
        "n_4_5": n_4_5, "n_6plus": n_6plus,
        "finish_dist_str": f"{n_win}-{n_2nd}-{n_3rd}-{n_4_5}-{n_6plus}",
        "win_rate": n_win / n,
        "place2_rate": n_p2 / n,
        "place3_rate": n_p3 / n,
        "avg_finish": float(fp.mean()) if n > 0 else None,
        "avg_popularity": float(pop.mean()) if pop is not None and pop.notna().any() else None,
        "win_roi": win_roi,
        "place_roi": place_roi,
        "place_bets": place_bets,
        "place_coverage": place_coverage,
    }


def _apply_conditions(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df
    df_dt = out["date"]
    if params.get("date_from"):
        out = out[df_dt >= pd.to_datetime(params["date_from"])]
    if params.get("date_to"):
        out = out[out["date"] <= pd.to_datetime(params["date_to"])]
    if params.get("venues"):
        vs = [v.strip() for v in str(params["venues"]).split(",") if v.strip()]
        if vs:
            out = out[out["venue"].astype(str).isin(vs)]
    if params.get("surfaces"):
        ss = [s.strip() for s in str(params["surfaces"]).split(",") if s.strip()]
        if ss:
            out = out[out["surface"].astype(str).isin(ss)]
    if params.get("dist_min") is not None:
        out = out[out["distance"] >= float(params["dist_min"])]
    if params.get("dist_max") is not None:
        out = out[out["distance"] <= float(params["dist_max"])]
    if params.get("track_conditions"):
        ts = [t.strip() for t in str(params["track_conditions"]).split(",") if t.strip()]
        if ts:
            out = out[out["track_condition"].astype(str).isin(ts)]
    if params.get("grades"):
        gs = [g.strip() for g in str(params["grades"]).split(",") if g.strip()]
        if gs:
            out = out[out["grade"].astype(str).isin(gs)]
    if params.get("only_finished"):
        out = out[out["finish_position"].notna() & (out["finish_position"] > 0)]
    return out


def compute_stats(
    target_kind: str,
    target_id: Optional[str] = None,
    main_group: Optional[str] = None,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    venues: Optional[str] = None,
    surfaces: Optional[str] = None,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    track_conditions: Optional[str] = None,
    grades: Optional[str] = None,
    breakdown: Optional[str] = None,
    top_per_breakdown: int = 30,
) -> dict[str, Any]:
    """指定した種牡馬 / クラスタ群について、期間 × 舞台条件で統計量を返す。

    breakdown: None | 'sire' | 'venue' | 'surface' | 'dist_cat' | 'grade' | 'L2' | 'L1_sub'
    """
    art = _load_stats_assets()
    if art is None:
        return {
            "status": "error",
            "message": (
                "stats アーティファクト未生成。`python -m src.research.pedigree.build_meta_cluster_artifacts` を実行してください。"
            ),
        }

    sids, label_map = _resolve_target_stallions(target_kind, target_id, main_group)
    if not sids:
        return {
            "status": "no_target",
            "message": f"対象 stallion が解決できません (kind={target_kind}, id={target_id}, mg={main_group})",
        }

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["stallion_id"].isin(sids)].copy()
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]
    params = dict(
        date_from=date_from, date_to=date_to,
        venues=venues, surfaces=surfaces,
        dist_min=dist_min, dist_max=dist_max,
        track_conditions=track_conditions, grades=grades,
        only_finished=True,
    )
    sub = _apply_conditions(sub, params)

    overall = _compute_aggregate(sub)

    breakdown_rows: list[dict[str, Any]] = []
    if breakdown:
        bk = breakdown.lower()
        unified = art["unified"]
        sid_to_main = art["sid_to_main"]

        if bk == "sire":
            for sid, g in sub.groupby("stallion_id"):
                a = _compute_aggregate(g)
                a["key"] = sid
                a["label"] = label_map.get(sid, sid)
                a["main_group"] = sid_to_main.get(sid)
                breakdown_rows.append(a)
        elif bk == "venue":
            for k, g in sub.groupby(sub["venue"].astype(str)):
                a = _compute_aggregate(g); a["key"] = k; a["label"] = k
                breakdown_rows.append(a)
        elif bk == "surface":
            for k, g in sub.groupby(sub["surface"].astype(str)):
                a = _compute_aggregate(g); a["key"] = k; a["label"] = k
                breakdown_rows.append(a)
        elif bk == "dist_cat":
            DIST_BINS_LOCAL = [0, 1400, 1800, 2400, 9999]
            DIST_LABELS_LOCAL = ["短距離", "マイル", "中距離", "長距離"]
            sub2 = sub.assign(
                dist_cat=pd.cut(sub["distance"], bins=DIST_BINS_LOCAL, labels=DIST_LABELS_LOCAL, right=False)
            )
            for k, g in sub2.groupby(sub2["dist_cat"].astype(str)):
                a = _compute_aggregate(g); a["key"] = k; a["label"] = k
                breakdown_rows.append(a)
        elif bk == "grade":
            for k, g in sub.groupby(sub["grade"].astype(str)):
                a = _compute_aggregate(g); a["key"] = k; a["label"] = k
                breakdown_rows.append(a)
        elif bk in ("l2", "l1_sub"):
            sid_to_l2: dict[str, int] = {}
            sid_to_l1: dict[str, int] = {}
            sid_to_mg: dict[str, str] = {}
            for _, r in unified.iterrows():
                ent = str(r["entity_id"])
                mg = str(r["main_group"])
                if mg == "非主流":
                    members = [s for s, root in art["root_of"].items() if str(root) == ent]
                else:
                    members = [ent]
                for m in members:
                    sid_to_l2[m] = int(r["L2"]) if int(r["L2"]) >= 0 else -1
                    sid_to_l1[m] = int(r["L1_sub"])
                    sid_to_mg[m] = mg
            sub2 = sub.copy()
            if bk == "l2":
                sub2["_grp"] = sub2["stallion_id"].map(sid_to_l2)
                for k, g in sub2.dropna(subset=["_grp"]).groupby("_grp"):
                    a = _compute_aggregate(g); a["key"] = int(k); a["label"] = f"L2={int(k)}"
                    breakdown_rows.append(a)
            else:
                sub2["_grp"] = sub2.apply(
                    lambda r: (sid_to_mg.get(r["stallion_id"]), sid_to_l1.get(r["stallion_id"])), axis=1
                )
                for k, g in sub2.dropna(subset=["_grp"]).groupby(sub2["_grp"].astype(str)):
                    sample_mg, sample_l1 = g["_grp"].iloc[0]
                    a = _compute_aggregate(g)
                    a["key"] = k
                    a["label"] = f"{sample_mg}/L1={sample_l1}"
                    a["main_group"] = sample_mg
                    breakdown_rows.append(a)

        breakdown_rows.sort(key=lambda x: x.get("n_records", 0), reverse=True)
        breakdown_rows = breakdown_rows[: max(1, int(top_per_breakdown))]

    return {
        "status": "ok",
        "target": {
            "kind": target_kind,
            "id": target_id,
            "main_group": main_group,
            "n_stallions": len(sids),
            "stallions_preview": [
                {"stallion_id": s, "stallion_name": label_map.get(s, s)}
                for s in sids[:20]
            ],
            "stallions_total": len(sids),
        },
        "filters": {
            "date_from": date_from, "date_to": date_to,
            "venues": venues, "surfaces": surfaces,
            "dist_min": dist_min, "dist_max": dist_max,
            "track_conditions": track_conditions, "grades": grades,
        },
        "overall": overall,
        "breakdown": {"key": breakdown, "rows": breakdown_rows} if breakdown else None,
    }


def compute_sire_presence_stats(
    stallion_id: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    venues: Optional[str] = None,
    surfaces: Optional[str] = None,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    track_conditions: Optional[str] = None,
    grades: Optional[str] = None,
) -> dict[str, Any]:
    """指定の種牡馬 X が血統 (直近10世代以内) のどの領域に含まれるかで層別した統計を返す。

    返却キー:
        - ``sire``    : X が父 (1代目) として登場する馬の成績 (= 既存 sire 統計と同じ)
        - ``father``  : X が父系領域 (父 / 父父 / 父父父 ... 父系全祖先 = 10gen 以内) に登場する馬の成績
                       「父父以降」のみではなく、A: 父以降領域 = X が父またはその上の父系祖先のどこかに登場
        - ``mother``  : X が母系領域 (母父系 OR 母母系 = 10gen 以内) に登場する馬の成績
        - ``father_only`` / ``mother_only`` / ``both`` : 重なりを排他的に分けた集計
    """
    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}
    art = _load_pedigree_10gen()
    if art is None or not art.get("ped_10gen_loaded"):
        return {"status": "error", "message": "10gen ancestor index 未生成"}

    sid = str(stallion_id)
    sid_to_name = art["sid_to_name"]
    name = sid_to_name.get(sid, sid)

    father_set: set[str] = art["ancestor_to_father_horses"].get(sid, set())
    mother_set: set[str] = art["ancestor_to_mother_horses"].get(sid, set())
    both_set: set[str] = father_set & mother_set
    father_only_set = father_set - both_set
    mother_only_set = mother_set - both_set

    rec: pd.DataFrame = art["records"]

    def _agg_for(horse_ids: set[str]) -> dict[str, Any]:
        if not horse_ids:
            return _compute_aggregate(rec.iloc[0:0])
        sub = rec[rec["horse_id"].isin(horse_ids)].copy()
        sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]
        params = dict(
            date_from=date_from, date_to=date_to,
            venues=venues, surfaces=surfaces,
            dist_min=dist_min, dist_max=dist_max,
            track_conditions=track_conditions, grades=grades,
            only_finished=True,
        )
        sub = _apply_conditions(sub, params)
        return _compute_aggregate(sub)

    sire_set = {h for h, s in art["sire_of_horse"].items() if s == sid}

    out = {
        "status": "ok",
        "stallion_id": sid,
        "stallion_name": name,
        "filters": {
            "date_from": date_from, "date_to": date_to,
            "venues": venues, "surfaces": surfaces,
            "dist_min": dist_min, "dist_max": dist_max,
            "track_conditions": track_conditions, "grades": grades,
            "scope": "直近10世代以内 (5gen × 5gen 再帰展開)",
        },
        "horse_counts": {
            "sire_total": len(sire_set),
            "father_total": len(father_set),
            "mother_total": len(mother_set),
            "both_total": len(both_set),
            "father_only_total": len(father_only_set),
            "mother_only_total": len(mother_only_set),
        },
        "sire": _agg_for(sire_set),
        "father": _agg_for(father_set),
        "mother": _agg_for(mother_set),
        "father_only": _agg_for(father_only_set),
        "mother_only": _agg_for(mother_only_set),
        "both": _agg_for(both_set),
    }
    return out


def compute_condition_ranking(
    *,
    side: str = "sire",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    venues: Optional[str] = None,
    surfaces: Optional[str] = None,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    track_conditions: Optional[str] = None,
    grades: Optional[str] = None,
    pop_min: Optional[float] = None,
    pop_max: Optional[float] = None,
    odds_min: Optional[float] = None,
    odds_max: Optional[float] = None,
    min_n: int = 30,
    top: int = 30,
    sort: str = "balanced",
) -> dict[str, Any]:
    """指定した馬券条件で「該当条件に強い種牡馬」のランキングを返す。

    side:
        'sire'   : 直結父 (1代直結)
        'bms'    : 母父 (2代目母父)
        'father' : 父系領域 (10gen) — 種牡馬 X が父・父父…のどこかにいる馬全部
        'mother' : 母系領域 (10gen) — 種牡馬 X が母父系 / 母母系のどこかにいる馬全部

    sort:
        'balanced'    : sqrt(win_roi × place_roi) × (1 + win_rate)  # 推奨 (デフォルト)
        'win_roi' / 'place_roi' / 'win_rate' / 'place3_rate' / 'n_records'
    """
    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}

    side_lc = (side or "sire").lower()
    if side_lc in ("father", "mother", "bms"):
        # 10gen 含有 or BMS が必要
        art = _load_pedigree_10gen()
        if art is None:
            return {"status": "error", "message": "10gen index 未生成"}

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["finish_position"].notna() & (rec["finish_position"] > 0)].copy()
    params = dict(
        date_from=date_from, date_to=date_to,
        venues=venues, surfaces=surfaces,
        dist_min=dist_min, dist_max=dist_max,
        track_conditions=track_conditions, grades=grades,
        only_finished=True,
    )
    sub = _apply_conditions(sub, params)
    if pop_min is not None and "popularity" in sub.columns:
        sub = sub[sub["popularity"].astype("Float64") >= float(pop_min)]
    if pop_max is not None and "popularity" in sub.columns:
        sub = sub[sub["popularity"].astype("Float64") <= float(pop_max)]
    if odds_min is not None and "odds" in sub.columns:
        sub = sub[sub["odds"].astype("Float64") >= float(odds_min)]
    if odds_max is not None and "odds" in sub.columns:
        sub = sub[sub["odds"].astype("Float64") <= float(odds_max)]
    # フィルタ条件をレスポンスに反映するため params に保存 (UI のチップ表示用)
    params["pop_min"] = pop_min
    params["pop_max"] = pop_max
    params["odds_min"] = odds_min
    params["odds_max"] = odds_max

    if sub.empty:
        return {
            "status": "ok",
            "side": side_lc,
            "filters": params,
            "min_n": min_n, "top": top,
            "sort": sort,
            "n_rows": 0,
            "n_records_after_filter": 0,
            "n_horses_after_filter": 0,
            "diagnose": "条件マッチするレース行が 0 件 (date/venue/surface/dist/track/grade/pop/odds の組合せでヒット 0)",
            "rows": [],
        }

    # 診断用: フィルタ後の生件数 (出走 / 馬数)
    n_records_after = int(len(sub))
    n_horses_after = int(sub["horse_id"].astype(str).nunique())

    sid_to_name = art["sid_to_name"]
    # 祖先 ID → 馬名 (10gen JSON 由来) を fallback マップとしても使う。
    # mother/father (10gen) や bms 側で sid_to_name に無い ID を解決するため。
    ancestor_name_map: dict[str, str] = art.get("ancestor_id_to_name", {}) or {}

    def _resolve_name(stallion_or_anc_id: str) -> str:
        sid_str = str(stallion_or_anc_id)
        nm = sid_to_name.get(sid_str)
        if nm and nm != sid_str:
            return nm
        nm2 = ancestor_name_map.get(sid_str)
        if nm2:
            return nm2
        return sid_str

    # side → 各馬の関連 stallion_ids
    if side_lc == "sire":
        sub["_stallion_id"] = sub["stallion_id"]
        groups = sub.dropna(subset=["_stallion_id"]).groupby("_stallion_id")
    elif side_lc == "bms":
        # horse_id -> bms_id を horse_bms.parquet 等から構築。ここでは race_records だけでは無理。
        # _load_artifacts() の中に存在する horse_to_bms か、別ソースが必要。
        h2b_path = IDX_DIR / "horse_bms.parquet"
        if not h2b_path.exists():
            return {"status": "error", "message": f"horse_bms.parquet が無い ({h2b_path})"}
        h2b = pd.read_parquet(h2b_path, columns=["horse_id", "bms_id"])
        h2b["horse_id"] = h2b["horse_id"].astype(str)
        h2b["bms_id"] = h2b["bms_id"].astype(str)
        bms_of: dict[str, str] = dict(zip(h2b["horse_id"], h2b["bms_id"]))
        sub["_stallion_id"] = sub["horse_id"].astype(str).map(bms_of)
        groups = sub.dropna(subset=["_stallion_id"]).groupby("_stallion_id")
    else:
        # father / mother: 各馬の祖先 ID 一覧を inverse して expanded long に
        # メモリ重いので、ancestor_to_horses の inverted map を使って ancestor 単位で集計
        side_key = "ancestor_to_father_horses" if side_lc == "father" else "ancestor_to_mother_horses"
        anc_map: dict[str, set[str]] = art.get(side_key, {})
        if not anc_map:
            return {"status": "error", "message": f"{side_key} が未ロード"}

        sub_idx = sub.set_index("horse_id", drop=False)
        rows_acc: list[dict[str, Any]] = []
        for ancestor_id, horse_set in anc_map.items():
            if not horse_set:
                continue
            mask_ids = list(horse_set & set(sub_idx.index))
            if not mask_ids:
                continue
            grp = sub_idx.loc[mask_ids]
            if len(grp) < min_n:
                continue
            a = _compute_aggregate(grp)
            if not a.get("n_records"):
                continue
            a["stallion_id"] = str(ancestor_id)
            a["stallion_name"] = _resolve_name(ancestor_id)
            rows_acc.append(a)

        return _condition_ranking_finalize(
            rows_acc, sort, top, min_n, side_lc, params,
            n_records_after, n_horses_after,
        )

    # sire / bms 共通: groupby で集計
    rows_acc = []
    n_groups_before = 0
    n_groups_after_minn = 0
    for sid, grp in groups:
        n_groups_before += 1
        if len(grp) < min_n:
            continue
        n_groups_after_minn += 1
        a = _compute_aggregate(grp)
        if not a.get("n_records"):
            continue
        a["stallion_id"] = str(sid)
        a["stallion_name"] = _resolve_name(sid)
        rows_acc.append(a)

    return _condition_ranking_finalize(
        rows_acc, sort, top, min_n, side_lc, params,
        n_records_after, n_horses_after,
        n_groups_before=n_groups_before, n_groups_after_minn=n_groups_after_minn,
    )


def _condition_ranking_finalize(
    rows: list[dict[str, Any]],
    sort: str,
    top: int,
    min_n: int,
    side_lc: str,
    params: dict[str, Any],
    n_records_after: int = 0,
    n_horses_after: int = 0,
    n_groups_before: Optional[int] = None,
    n_groups_after_minn: Optional[int] = None,
) -> dict[str, Any]:
    """compute_condition_ranking の最終処理 (スコアリング・ソート・タグ付与)。"""
    import math

    art = _load_artifacts()
    full_tags = _load_full_stallion_tags() or {}
    sire_tags = (art or {}).get("sire_tags", {}) if art else {}
    tag_meta = (art or {}).get("tag_meta", {}) if art else {}

    # ポップアップで詳細統計を出すため、tag dict にある拡張統計を全部透過させる
    _STAT_KEYS = (
        "rate_eb", "rate_raw", "lift", "n_records", "wins",
        "baseline_eb", "cond_global", "share", "sire_n_total",
        "is_top20", "is_strong_abs", "is_relative_strong", "is_specialist",
    )

    def _passthrough_stats(t: dict, dst: dict) -> dict:
        for k in _STAT_KEYS:
            if k in t and t[k] is not None:
                dst[k] = t[k]
        return dst

    def _norm_tags(sid: str) -> list[dict[str, Any]]:
        full = full_tags.get(sid)
        if full:
            return [
                _passthrough_stats(t, {
                    "tag_id": t.get("tag_id"),
                    "label": t.get("tag_label") or t.get("label") or t.get("tag_id"),
                    "tag_label": t.get("tag_label") or t.get("label") or t.get("tag_id"),
                    "category": t.get("category", ""),
                    "color": t.get("category_color", "#888"),
                })
                for t in full
            ][:6]
        # strict tags は str / dict の両方の場合があり得る
        strict = sire_tags.get(sid) or []
        out: list[dict[str, Any]] = []
        for t in strict[:6]:
            if isinstance(t, dict):
                cat = t.get("category", "")
                base = {
                    "tag_id": t.get("tag_id"),
                    "label": t.get("tag_label") or t.get("label") or t.get("tag_id"),
                    "tag_label": t.get("tag_label") or t.get("label") or t.get("tag_id"),
                    "category": cat,
                    "color": t.get("category_color") or TAG_CATEGORY_COLORS.get(cat, "#888"),
                }
                _passthrough_stats(t, base)
                out.append(base)
            else:
                cat = tag_meta.get(t, {}).get("category", "")
                out.append({
                    "tag_id": t,
                    "label": tag_meta.get(t, {}).get("label", t),
                    "tag_label": tag_meta.get(t, {}).get("label", t),
                    "category": cat,
                    "color": TAG_CATEGORY_COLORS.get(cat, "#888"),
                })
        return out

    for r in rows:
        win_roi   = r.get("win_roi") or 0.0
        place_roi = r.get("place_roi") or 0.0
        win_rate  = r.get("win_rate") or 0.0
        if win_roi > 0 and place_roi > 0:
            r["balanced_score"] = math.sqrt(win_roi * place_roi) * (1.0 + win_rate)
        else:
            r["balanced_score"] = 0.0
        sid = str(r["stallion_id"])
        r["tags"] = _norm_tags(sid)

    SORT_KEYS = {
        "balanced":     ("balanced_score", True),
        "win_roi":      ("win_roi", True),
        "place_roi":    ("place_roi", True),
        "win_rate":     ("win_rate", True),
        "place3_rate":  ("place3_rate", True),
        "place2_rate":  ("place2_rate", True),
        "n_records":    ("n_records", True),
    }
    key, desc = SORT_KEYS.get(sort, ("balanced_score", True))
    rows.sort(key=lambda r: (r.get(key) or 0), reverse=desc)
    out_rows = rows[: top]

    diagnose = None
    if not rows:
        if n_records_after == 0:
            diagnose = "条件マッチするレース行が 0 件 (フィルタの組合せが厳しすぎ、または矛盾)"
        elif n_groups_before is not None and n_groups_before == 0:
            diagnose = f"レース行は {n_records_after:,} 件マッチしたが、対象 stallion ({side_lc}) に紐付かない"
        elif n_groups_after_minn is not None and n_groups_after_minn == 0:
            diagnose = (
                f"レース行は {n_records_after:,} 件 / 対象集団 {n_groups_before} グループあるが、"
                f"min_n={min_n} を満たすグループが 0 件。min_n を下げてください。"
            )
        else:
            diagnose = f"min_n={min_n} を満たすグループが 0 件"

    return {
        "status": "ok",
        "side": side_lc,
        "filters": params,
        "min_n": min_n,
        "top": top,
        "sort": sort,
        "n_rows": len(rows),
        "n_records_after_filter": n_records_after,
        "n_horses_after_filter": n_horses_after,
        "n_groups_before_minn": n_groups_before,
        "n_groups_after_minn": n_groups_after_minn,
        "diagnose": diagnose,
        "rows": out_rows,
    }


def compute_progeny_under_condition(
    *,
    stallion_id: str,
    side: str = "sire",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    venues: Optional[str] = None,
    surfaces: Optional[str] = None,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    track_conditions: Optional[str] = None,
    grades: Optional[str] = None,
    pop_min: Optional[float] = None,
    pop_max: Optional[float] = None,
    odds_min: Optional[float] = None,
    odds_max: Optional[float] = None,
    min_n: int = 3,
    top: int = 12,
    sort: str = "balanced",
    sample_per_horse: int = 5,
) -> dict[str, Any]:
    """指定 stallion_id (種牡馬 / BMS / 父系祖先 / 母系祖先) の該当馬を、
    与えられた馬券条件で集計し、代表的な該当馬の条件成績を返す。

    探索モードのテーブル行展開 (折り畳み詳細) 用。条件は compute_condition_ranking
    と完全に同じものを受け付ける。
    """
    sid = str(stallion_id)

    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}

    side_lc = (side or "sire").lower()
    if side_lc in ("father", "mother", "bms"):
        art = _load_pedigree_10gen()
        if art is None:
            return {"status": "error", "message": "10gen index 未生成"}

    # 名前マスタ読み込み (馬名表示のため)
    art_names = _load_horse_prize_map() or art
    name_map: dict[str, str] = (art_names or {}).get("horse_name_map", {})
    sid_to_name = art.get("sid_to_name", {})
    stallion_name = sid_to_name.get(sid, sid) if isinstance(sid_to_name, dict) else sid

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["finish_position"].notna() & (rec["finish_position"] > 0)].copy()
    params = dict(
        date_from=date_from, date_to=date_to,
        venues=venues, surfaces=surfaces,
        dist_min=dist_min, dist_max=dist_max,
        track_conditions=track_conditions, grades=grades,
        only_finished=True,
    )
    sub = _apply_conditions(sub, params)
    if pop_min is not None and "popularity" in sub.columns:
        sub = sub[sub["popularity"].astype("Float64") >= float(pop_min)]
    if pop_max is not None and "popularity" in sub.columns:
        sub = sub[sub["popularity"].astype("Float64") <= float(pop_max)]
    if odds_min is not None and "odds" in sub.columns:
        sub = sub[sub["odds"].astype("Float64") >= float(odds_min)]
    if odds_max is not None and "odds" in sub.columns:
        sub = sub[sub["odds"].astype("Float64") <= float(odds_max)]
    params["pop_min"] = pop_min
    params["pop_max"] = pop_max
    params["odds_min"] = odds_min
    params["odds_max"] = odds_max

    if sub.empty:
        return {"status": "ok", "stallion_id": sid, "stallion_name": stallion_name,
                "side": side_lc, "filters": params, "n_horses_total": 0,
                "n_records_total": 0, "rows": []}

    # side ごとに該当馬集合を絞り込み
    if side_lc == "sire":
        target = sub[sub["stallion_id"].astype(str) == sid]
    elif side_lc == "bms":
        h2b_path = IDX_DIR / "horse_bms.parquet"
        if not h2b_path.exists():
            return {"status": "error", "message": f"horse_bms.parquet が無い ({h2b_path})"}
        h2b = pd.read_parquet(h2b_path, columns=["horse_id", "bms_id"])
        h2b = h2b.dropna(subset=["bms_id"])
        h2b["horse_id"] = h2b["horse_id"].astype(str)
        h2b["bms_id"] = h2b["bms_id"].astype(str)
        horses = set(h2b.loc[h2b["bms_id"] == sid, "horse_id"])
        if not horses:
            target = sub.iloc[0:0]
        else:
            target = sub[sub["horse_id"].astype(str).isin(horses)]
    else:
        side_key = "ancestor_to_father_horses" if side_lc == "father" else "ancestor_to_mother_horses"
        anc_map: dict[str, set[str]] = art.get(side_key, {}) or {}
        horses = anc_map.get(sid, set())
        if not horses:
            target = sub.iloc[0:0]
        else:
            target = sub[sub["horse_id"].astype(str).isin(horses)]

    if target.empty:
        return {"status": "ok", "stallion_id": sid, "stallion_name": stallion_name,
                "side": side_lc, "filters": params, "n_horses_total": 0,
                "n_records_total": 0, "rows": []}

    # 全体サマリー
    n_records_total = int(len(target))
    n_horses_total = int(target["horse_id"].nunique())

    # 馬ごとに集計
    rows_acc: list[dict[str, Any]] = []
    for hid, grp in target.groupby("horse_id"):
        if len(grp) < min_n:
            continue
        a = _compute_aggregate(grp)
        if not a.get("n_records"):
            continue
        a["horse_id"] = str(hid)
        a["horse_name"] = name_map.get(str(hid), str(hid))
        # balanced score (ROI が出ない場合は win_rate ベースで補完)
        wr = a.get("win_rate") or 0.0
        wroi = a.get("win_roi") or 0.0
        proi = a.get("place_roi") or 0.0
        if wroi > 0 and proi > 0:
            import math as _m
            a["balanced_score"] = _m.sqrt(wroi * proi) * (1.0 + wr)
        else:
            a["balanced_score"] = wr * 100.0  # ROI 未計算時の代替
        # 平均着 < 平均人気 → 期待を上回るパフォーマンス
        af = a.get("avg_finish")
        ap = a.get("avg_popularity")
        a["upset_gap"] = (float(ap) - float(af)) if (af is not None and ap is not None) else None
        rows_acc.append(a)

    SORT_KEYS = {
        "balanced":     ("balanced_score", True),
        "win_rate":     ("win_rate", True),
        "place3_rate":  ("place3_rate", True),
        "win_roi":      ("win_roi", True),
        "place_roi":    ("place_roi", True),
        "n_records":    ("n_records", True),
        "upset_gap":    ("upset_gap", True),
    }
    key, desc = SORT_KEYS.get(sort, ("balanced_score", True))
    rows_acc.sort(key=lambda r: (r.get(key) if r.get(key) is not None else -1e18), reverse=desc)
    out_rows = rows_acc[: top]

    # 各馬の代表的なレース (該当条件下) を最新日付順に sample_per_horse 件
    out_horse_ids = [r["horse_id"] for r in out_rows]
    if out_horse_ids and sample_per_horse > 0:
        sub_idx = target[target["horse_id"].astype(str).isin(out_horse_ids)].copy()
        if "date" in sub_idx.columns:
            sub_idx = sub_idx.sort_values("date", ascending=False)
        cols_keep = [c for c in [
            "race_id", "date", "venue", "surface", "distance",
            "track_condition", "grade", "popularity", "odds", "finish_position",
            "field_size", "place_payout_yen",
        ] if c in sub_idx.columns]
        import math as _math
        sample_by_horse: dict[str, list[dict[str, Any]]] = {}
        for hid, g in sub_idx.groupby("horse_id"):
            recs = g[cols_keep].head(sample_per_horse).to_dict("records")
            cleaned: list[dict[str, Any]] = []
            for r in recs:
                # 日付フォーマット
                d = r.get("date")
                if d is not None and hasattr(d, "strftime"):
                    r["date"] = d.strftime("%Y-%m-%d")
                # NaN -> None & 数値正規化 (JSON 準拠)
                for k in list(r.keys()):
                    v = r[k]
                    if v is None:
                        continue
                    if isinstance(v, float) and _math.isnan(v):
                        r[k] = None
                        continue
                    if k in ("popularity", "odds", "finish_position", "field_size", "distance", "place_payout_yen"):
                        if isinstance(v, (str, bool)):
                            continue
                        try:
                            f = float(v)
                            if _math.isnan(f):
                                r[k] = None
                            elif k in ("popularity", "finish_position", "field_size", "distance"):
                                r[k] = int(f)
                            else:
                                r[k] = f
                        except (ValueError, TypeError):
                            r[k] = None
                cleaned.append(r)
            sample_by_horse[str(hid)] = cleaned
        for r in out_rows:
            r["sample_races"] = sample_by_horse.get(r["horse_id"], [])

    return {
        "status": "ok",
        "stallion_id": sid,
        "stallion_name": stallion_name,
        "side": side_lc,
        "filters": params,
        "min_n": min_n,
        "top": top,
        "sort": sort,
        "n_horses_total": n_horses_total,
        "n_records_total": n_records_total,
        "n_horses_matched": len(rows_acc),
        "rows": out_rows,
    }


def compute_sire_best_conditions(
    stallion_id: str,
    *,
    min_n: int = 30,
    top: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict[str, Any]:
    """指定種牡馬 X 直結産駒 (sire) の出走を、複数の条件軸で集計してランキングを返す。

    出力:
      rankings.combo (venue × surface × dist_cat)
         by_win_rate / by_place3_rate / by_win_roi / by_place_roi
      rankings.combo_full (venue × surface × dist_cat × track_condition × grade)
         by_win_roi / by_place_roi  (より粒度の細かい複合)
      rankings.by_venue / by_surface / by_dist_cat / by_track_condition / by_grade
         単軸別の全行 (n_records 降順) - フロントでソート切替

    KPI: 勝率 / 連対率 / 複勝率 / 平均着 / 平均人気 / 単回収 / 複回収 / n_records
    """
    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}

    sid = str(stallion_id)
    sid_to_name = art["sid_to_name"]
    name = sid_to_name.get(sid, sid)

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["stallion_id"] == sid].copy()
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]

    params = dict(
        date_from=date_from, date_to=date_to,
        only_finished=True,
    )
    sub = _apply_conditions(sub, params)

    if sub.empty:
        return {
            "status": "ok",
            "stallion_id": sid,
            "stallion_name": name,
            "min_n": int(min_n),
            "top": int(top),
            "n_total": 0,
            "rankings": {},
        }

    DIST_BINS_LOCAL = [0, 1400, 1800, 2400, 9999]
    DIST_LABELS_LOCAL = ["短距離", "マイル", "中距離", "長距離"]
    sub = sub.assign(
        dist_cat=pd.cut(sub["distance"], bins=DIST_BINS_LOCAL, labels=DIST_LABELS_LOCAL, right=False).astype(str),
        venue=sub["venue"].astype(str).fillna(""),
        surface=sub["surface"].astype(str).fillna(""),
        track_condition=sub["track_condition"].astype(str).fillna(""),
        grade=sub["grade"].astype(str).fillna(""),
    )
    # 欠損 / NaN / 空文字を含む行は集計から除外 (== 'nan' は astype(str) で発生)
    _BAD = {"nan", "None", "", "NaT"}
    for c in ("venue", "surface", "dist_cat", "track_condition", "grade"):
        sub = sub[~sub[c].isin(_BAD)]
    if sub.empty:
        return {
            "status": "ok",
            "stallion_id": sid,
            "stallion_name": name,
            "min_n": int(min_n),
            "top": int(top),
            "n_total": 0,
            "rankings": {},
        }

    def _rank_by(rows: list[dict[str, Any]], key: str, *, reverse=True, n: int = None) -> list[dict[str, Any]]:
        items = [r for r in rows if r.get(key) is not None]
        items.sort(key=lambda r: r[key], reverse=reverse)
        if n is None:
            return items
        return items[:n]

    combo_rows = _build_groupby_fast(
        sub, ["venue", "surface", "dist_cat"],
        lambda keys: f"{keys[0]} × {keys[1]} × {keys[2]}", min_n=min_n
    )
    combo_full_rows = _build_groupby_fast(
        sub, ["venue", "surface", "dist_cat", "track_condition", "grade"],
        lambda keys: f"{keys[0]} × {keys[1]} × {keys[2]} × {keys[3]} × {keys[4]}", min_n=min_n
    )
    by_venue_rows   = _build_groupby_fast(sub, ["venue"],           lambda k: str(k[0]), min_n=min_n)
    by_surface_rows = _build_groupby_fast(sub, ["surface"],         lambda k: str(k[0]), min_n=min_n)
    by_dist_rows    = _build_groupby_fast(sub, ["dist_cat"],        lambda k: str(k[0]), min_n=min_n)
    by_track_rows   = _build_groupby_fast(sub, ["track_condition"], lambda k: str(k[0]), min_n=min_n)
    by_grade_rows   = _build_groupby_fast(sub, ["grade"],           lambda k: str(k[0]), min_n=min_n)

    return {
        "status": "ok",
        "stallion_id": sid,
        "stallion_name": name,
        "min_n": int(min_n),
        "top": int(top),
        "n_total": int(len(sub)),
        "filters": {
            "date_from": date_from,
            "date_to": date_to,
        },
        "rankings": {
            "combo": {
                "axis": "場 × 路面 × 距離区分",
                "all_rows": combo_rows,
                "by_win_rate":   _rank_by(combo_rows, "win_rate",    reverse=True, n=top),
                "by_place3_rate":_rank_by(combo_rows, "place3_rate", reverse=True, n=top),
                "by_win_roi":    _rank_by(combo_rows, "win_roi",     reverse=True, n=top),
                "by_place_roi":  _rank_by(combo_rows, "place_roi",   reverse=True, n=top),
            },
            "combo_full": {
                "axis": "場 × 路面 × 距離区分 × 馬場 × グレード",
                "all_rows": combo_full_rows,
                "by_win_roi":    _rank_by(combo_full_rows, "win_roi",   reverse=True, n=top),
                "by_place_roi":  _rank_by(combo_full_rows, "place_roi", reverse=True, n=top),
            },
            "by_venue":           {"axis": "競馬場", "all_rows": by_venue_rows},
            "by_surface":         {"axis": "路面",   "all_rows": by_surface_rows},
            "by_dist_cat":        {"axis": "距離区分","all_rows": by_dist_rows},
            "by_track_condition": {"axis": "馬場",   "all_rows": by_track_rows},
            "by_grade":           {"axis": "グレード","all_rows": by_grade_rows},
        },
    }


def compute_sire_heatmap(
    stallion_id: str,
    *,
    axis_x: str = "venue",
    axis_y: str = "dist_cat",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_n_cell: int = 1,
    surfaces: Optional[str] = None,
    track_conditions: Optional[str] = None,
) -> dict[str, Any]:
    """指定種牡馬 直結産駒の 2 軸ヒートマップ用集計を返す。

    axis_x / axis_y: 'venue' | 'dist_cat' | 'surface' | 'track_condition' | 'grade'
    各セル: n_records, win_rate, place3_rate, win_roi, place_roi, n_win
    """
    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}

    sid = str(stallion_id)
    sid_to_name = art["sid_to_name"]
    name = sid_to_name.get(sid, sid)

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["stallion_id"] == sid].copy()
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]
    sub = _apply_conditions(sub, dict(
        date_from=date_from, date_to=date_to,
        surfaces=surfaces, track_conditions=track_conditions,
        only_finished=True,
    ))
    if sub.empty:
        return {"status": "ok", "stallion_id": sid, "stallion_name": name,
                "axis_x": axis_x, "axis_y": axis_y,
                "x_labels": [], "y_labels": [], "cells": [], "n_total": 0}

    DIST_BINS_LOCAL = [0, 1400, 1800, 2400, 9999]
    DIST_LABELS_LOCAL = ["短距離", "マイル", "中距離", "長距離"]
    if "dist_cat" not in sub.columns:
        sub["dist_cat"] = pd.cut(sub["distance"], bins=DIST_BINS_LOCAL, labels=DIST_LABELS_LOCAL, right=False).astype(str)
    for c in ("venue", "surface", "track_condition", "grade"):
        if c in sub.columns:
            sub[c] = sub[c].astype(str).fillna("")
    _BAD = {"nan", "None", "", "NaT"}

    # ラベル順序 (見やすさのため固定)
    LABEL_ORDER = {
        "venue":           ["東京", "中山", "阪神", "京都", "中京", "新潟", "福島", "小倉", "函館", "札幌"],
        "dist_cat":        ["短距離", "マイル", "中距離", "長距離"],
        "surface":         ["芝", "ダート"],
        "track_condition": ["良", "稍重", "重", "不良"],
        "grade":           ["G1", "G2", "G3", "L", "OP", "3勝", "2勝", "1勝", "未勝利", "新馬"],
    }

    def _labels_for(axis_key: str, present_set: set[str]) -> list[str]:
        order = LABEL_ORDER.get(axis_key, [])
        labels = [v for v in order if v in present_set]
        # 順序外で存在するものを末尾に
        for v in sorted(present_set):
            if v and v not in labels and v not in _BAD:
                labels.append(v)
        return labels

    sub = sub[~sub[axis_x].astype(str).isin(_BAD)]
    sub = sub[~sub[axis_y].astype(str).isin(_BAD)]

    x_set = set(sub[axis_x].astype(str).unique())
    y_set = set(sub[axis_y].astype(str).unique())
    x_labels = _labels_for(axis_x, x_set)
    y_labels = _labels_for(axis_y, y_set)

    cells: list[dict[str, Any]] = []
    for (xv, yv), g in sub.groupby([axis_x, axis_y]):
        if len(g) < min_n_cell:
            continue
        a = _compute_aggregate(g)
        cells.append({
            "x": str(xv),
            "y": str(yv),
            "n_records": a["n_records"],
            "n_horses": a["n_horses"],
            "n_win": a["n_win"],
            "win_rate": a["win_rate"],
            "place2_rate": a["place2_rate"],
            "place3_rate": a["place3_rate"],
            "win_roi": a["win_roi"],
            "place_roi": a["place_roi"],
        })

    return {
        "status": "ok",
        "stallion_id": sid,
        "stallion_name": name,
        "axis_x": axis_x,
        "axis_y": axis_y,
        "n_total": int(len(sub)),
        "x_labels": x_labels,
        "y_labels": y_labels,
        "cells": cells,
        "filters": {"date_from": date_from, "date_to": date_to,
                    "surfaces": surfaces, "track_conditions": track_conditions},
    }


def compute_sire_summary_card(
    stallion_id: str,
    *,
    period: str = "all",
) -> dict[str, Any]:
    """種牡馬サマリーカード用の集計 (上部ヘッダ用)。

    period: 'all' | '2y' | '1y'
    返却: 主要 KPI + L2 + main_group + タグ
    """
    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}

    sid = str(stallion_id)
    sid_to_name = art["sid_to_name"]
    name = sid_to_name.get(sid, sid)

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["stallion_id"] == sid].copy()
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]

    today = pd.Timestamp.now()
    date_from: Optional[str] = None
    if period == "1y":
        date_from = (today - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    elif period == "2y":
        date_from = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    if date_from:
        sub = sub[sub["date"] >= pd.to_datetime(date_from)]

    overall = _compute_aggregate(sub)

    # L2 + main_group + tags
    unified = art["unified"]
    unified_by_entity = art["unified_by_entity"]
    sid_to_main = art["sid_to_main"]
    L2 = None
    main_group = sid_to_main.get(sid)
    if sid in unified_by_entity.index:
        row = unified_by_entity.loc[sid]
        try:
            L2 = int(row["L2"])
        except Exception:
            L2 = None

    full_tags = _load_full_stallion_tags() or {}
    sire_tags = art.get("sire_tags", {}) or {}
    tag_meta = art.get("tag_meta", {}) or {}
    tags_out: list[dict[str, Any]] = []
    if full_tags.get(sid):
        for t in full_tags[sid]:
            tags_out.append({
                "tag_id": t.get("tag_id"),
                "label": t.get("tag_label") or t.get("label") or t.get("tag_id"),
                "category": t.get("category", ""),
                "color": t.get("category_color", "#888"),
            })
    else:
        for t in (sire_tags.get(sid) or []):
            cat = tag_meta.get(t, {}).get("category", "")
            tags_out.append({
                "tag_id": t,
                "label": tag_meta.get(t, {}).get("label", t),
                "category": cat,
                "color": TAG_CATEGORY_COLORS.get(cat, "#888"),
            })

    return {
        "status": "ok",
        "stallion_id": sid,
        "stallion_name": name,
        "period": period,
        "date_from": date_from,
        "L2": L2,
        "L2_meta": get_l2_meta(L2),
        "main_group": main_group,
        "overall": overall,
        "tags": tags_out,
    }


def compute_sire_presence_horses(
    stallion_id: str,
    side: str = "father",
    *,
    limit: int = 10,
    offset: int = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    venues: Optional[str] = None,
    surfaces: Optional[str] = None,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    track_conditions: Optional[str] = None,
    grades: Optional[str] = None,
) -> dict[str, Any]:
    """指定種牡馬 X を ``side`` (father=A=父系領域 / mother=B=母系領域 / sire=父=本馬 / both) に
    持つ馬の一覧を、推定獲得賞金 (本賞金 1-5 着換算) の多い順で返す。

    フィルタ条件 (期間/場/路面/距離/馬場/グレード) を適用した出走のみで賞金を集計する。
    """
    art = _load_stats_assets()
    if art is None:
        return {"status": "error", "message": "stats アーティファクト未生成"}
    art = _load_pedigree_10gen()
    if art is None or not art.get("ped_10gen_loaded"):
        return {"status": "error", "message": "10gen ancestor index 未生成"}
    art = _load_horse_prize_map()
    if art is None:
        return {"status": "error", "message": "horse prize map 構築失敗"}

    sid = str(stallion_id)
    sid_to_name = art["sid_to_name"]
    ancestor_name_map: dict[str, str] = art.get("ancestor_id_to_name", {}) or {}
    stallion_name = sid_to_name.get(sid) or ancestor_name_map.get(sid) or sid

    side_lc = (side or "father").lower()
    father_set: set[str] = art["ancestor_to_father_horses"].get(sid, set())
    mother_set: set[str] = art["ancestor_to_mother_horses"].get(sid, set())
    sire_set: set[str] = {h for h, s in art["sire_of_horse"].items() if s == sid}
    both_set: set[str] = father_set & mother_set

    if side_lc == "father":
        target_set = father_set
        side_label = "A: 父系領域"
    elif side_lc == "mother":
        target_set = mother_set
        side_label = "B: 母系領域"
    elif side_lc == "sire":
        target_set = sire_set
        side_label = "★ 父=本馬"
    elif side_lc == "both":
        target_set = both_set
        side_label = "両方 (インブリード)"
    else:
        return {"status": "error", "message": f"unknown side: {side}"}

    rec: pd.DataFrame = art["records"]
    sub = rec[rec["horse_id"].isin(target_set)].copy()
    sub = sub[sub["finish_position"].notna() & (sub["finish_position"] > 0)]
    params = dict(
        date_from=date_from, date_to=date_to,
        venues=venues, surfaces=surfaces,
        dist_min=dist_min, dist_max=dist_max,
        track_conditions=track_conditions, grades=grades,
        only_finished=True,
    )
    sub = _apply_conditions(sub, params)

    if sub.empty:
        return {
            "status": "ok",
            "stallion_id": sid,
            "stallion_name": stallion_name,
            "side": side_lc,
            "side_label": side_label,
            "n_total_horses": 0,
            "n_filtered_horses": 0,
            "limit": limit,
            "offset": offset,
            "horses": [],
        }

    sub_top = sub[sub["finish_position"].astype(int) <= 5].copy()
    sub_top["_prize"] = [
        _estimate_prize_per_record(g, f)
        for g, f in zip(sub_top["grade"].astype(str), sub_top["finish_position"])
    ]
    prize_per = sub_top.groupby("horse_id")["_prize"].sum()
    counts_per = sub.groupby("horse_id")["finish_position"].agg(["count",
                  lambda s: int((s == 1).sum()),
                  lambda s: int((s == 2).sum()),
                  lambda s: int((s == 3).sum())])
    counts_per.columns = ["n_records", "n_win", "n_p2", "n_p3"]
    avg_finish = sub.groupby("horse_id")["finish_position"].mean().rename("avg_finish")
    df_horse = counts_per.join(avg_finish, how="left")
    df_horse["_prize_filtered"] = prize_per.reindex(df_horse.index).fillna(0.0)

    name_map: dict[str, str] = art["horse_name_map"]

    df_horse["horse_id"] = df_horse.index
    df_horse["horse_name"] = df_horse["horse_id"].map(lambda h: name_map.get(h, ""))

    full_prize_map: dict[str, float] = art["horse_prize_man"]
    df_horse["_prize_career"] = df_horse["horse_id"].map(lambda h: float(full_prize_map.get(h, 0.0)))

    df_horse = df_horse.sort_values(
        by=["_prize_career", "_prize_filtered", "n_win"],
        ascending=[False, False, False],
    )

    n_total = len(target_set)
    n_filtered = int(len(df_horse))
    page = df_horse.iloc[offset: offset + limit] if limit > 0 else df_horse.iloc[offset:]

    horses_out: list[dict[str, Any]] = []
    for _, r in page.iterrows():
        horses_out.append({
            "horse_id": str(r["horse_id"]),
            "horse_name": r["horse_name"] or str(r["horse_id"]),
            "career_prize_man": float(r["_prize_career"]),
            "filtered_prize_man": float(r["_prize_filtered"]),
            "n_records": int(r["n_records"]),
            "n_win": int(r["n_win"]),
            "n_p2": int(r["n_p2"]),
            "n_p3": int(r["n_p3"]),
            "avg_finish": float(r["avg_finish"]) if pd.notna(r["avg_finish"]) else None,
            "netkeiba_url": f"https://db.netkeiba.com/horse/{r['horse_id']}/",
        })

    return {
        "status": "ok",
        "stallion_id": sid,
        "stallion_name": stallion_name,
        "side": side_lc,
        "side_label": side_label,
        "n_total_horses": n_total,
        "n_filtered_horses": n_filtered,
        "limit": limit,
        "offset": offset,
        "filters": {
            "date_from": date_from, "date_to": date_to,
            "venues": venues, "surfaces": surfaces,
            "dist_min": dist_min, "dist_max": dist_max,
            "track_conditions": track_conditions, "grades": grades,
        },
        "horses": horses_out,
    }
