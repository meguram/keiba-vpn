"""血統 role 別 lift プロファイルを集計するバッチ。

ある種牡馬 X が「父 (F)」「母父 (MF)」「母母父 (MMF)」「父父 (FF)」のどの役割で登場する
馬の集合（= その馬たちのレース戦績）から、条件別 EB 勝率と「自分の eb_total での lift」
を計算し、`role_lift_profiles.parquet` として保存する。

検証スクリプト `src/research/pedigree/role_lift_evidence.py` の結果（H1〜H4）に基づく:
  - F: 安定（n大・std小）→ 主軸
  - MF: F と相関低く独立 → 第二の重み
  - FF: 父父系統。長距離・芝で有意
  - MMF: 隠れ適性シグナル

使用:
  python -m src.research.pedigree.build_role_lift_profiles
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger("research.role_lift")

ROOT = Path(__file__).resolve().parents[3]
IDX_DIR = ROOT / "data/research/pedigree_race_index"
ART_DIR = ROOT / "data/research/bloodline_meta_cluster"

# Empirical Bayes prior （特殊適性タグと整合）
EB_PRIOR_N = 30
# 条件別の global mean は overall_raw.json から取得（fallback あり）

# 距離区分は父視点 (build_meta_cluster_artifacts.py) と統一:
#   短距離 [0,1400) / マイル [1400,1800) / 中距離 [1800,2400) / 長距離 [2400,9999)
DIST_BINS = [0, 1400, 1800, 2400, 9999]
DIST_LABELS = ["短距離", "マイル", "中距離", "長距離"]
STEEP_VENUES = ("中山", "阪神", "中京")
COND_COLS = (
    [f"win_v_{v}" for v in ["東京", "中山", "阪神", "京都", "中京", "新潟", "小倉", "福島", "札幌", "函館"]]
    + [f"win_s_{s}" for s in ["芝", "ダート"]]
    + [f"win_d_{d}" for d in DIST_LABELS]
    + ["win_steep", "win_flat", "win_heavy", "win_outer", "win_inner"]
)

# role の定義
#   F   : sire_id 直接（horse_bms.parquet）
#   MF  : bms_id 直接（horse_bms.parquet）
#   MMF : path_fm == 'MMF' （horse_pedigree_cats.parquet）
#   FF  : path_fm == 'FF'
ROLES = ("F", "MF", "MMF", "FF")
MIN_N_RECORDS = 30  # この出走数未満の (stallion, role) は除外


def _eb(w: float, n: int, prior_p: float) -> float | None:
    """EB shrinkage. prior は条件別 global mean。"""
    if n < 3:
        return None
    return (w + EB_PRIOR_N * prior_p) / (n + EB_PRIOR_N)


def _load_overall_priors() -> dict[str, float]:
    """overall_raw.json (条件別の全種牡馬平均勝率) をロードして prior として使う。"""
    path = ART_DIR / "overall_raw.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("overall_raw.json 読込失敗 (%s) → fallback 0.139 を使う", e)
    return {c: 0.139 for c in COND_COLS}


def _build_role_indices(
    cats: pd.DataFrame, bms: pd.DataFrame
) -> dict[str, dict[str, set[str]]]:
    """role -> stallion_id -> set(horse_id)"""
    idx: dict[str, dict[str, set[str]]] = {}
    idx["F"] = bms.groupby("sire_id")["horse_id"].apply(set).to_dict()
    idx["MF"] = bms.groupby("bms_id")["horse_id"].apply(set).to_dict()
    for path in ("MMF", "FF"):
        sub = cats[cats["path_fm"] == path]
        idx[path] = sub.groupby("stallion_id")["horse_id"].apply(set).to_dict()
    return idx


def _resolve_target_stallions(
    cats: pd.DataFrame, bms: pd.DataFrame
) -> tuple[set[str], dict[str, str]]:
    """集計対象の stallion_id を返す。
    ・unified.parquet にある主流種牡馬 (133 頭)
    ・sid_to_main.json にある主流系統候補 (約 4,773 頭) のうち n_records が一定以上の母父系種牡馬

    name マップは sid_to_name.json + sid_to_name_full.json を結合。
    """
    sid_to_main: dict[str, str] = json.loads(
        (ART_DIR / "sid_to_main.json").read_text(encoding="utf-8")
    )
    sid_to_name: dict[str, str] = {}
    for fname in ("sid_to_name_full.json", "sid_to_name.json"):
        p = ART_DIR / fname
        if p.exists():
            try:
                cur = json.loads(p.read_text(encoding="utf-8"))
                # full 優先で sid_to_name を更新
                sid_to_name.update(cur)
            except Exception:
                pass

    # 主流候補: sid_to_main.json の全 ID（4,773 頭程度）
    candidates: set[str] = set(sid_to_main.keys())

    # 加えて、unified.parquet にいる主流種牡馬
    unified = pd.read_parquet(ART_DIR / "unified.parquet", columns=["entity_id", "entity_type"])
    candidates |= set(
        unified[unified["entity_type"] == "stallion"]["entity_id"].astype(str)
    )

    # 母父・父父・母母父として登場する候補 (cats の stallion_id)
    candidates |= set(cats["stallion_id"].astype(str).unique())
    candidates |= set(bms["sire_id"].astype(str).unique())
    candidates |= set(bms["bms_id"].astype(str).unique())

    # サイズが大きすぎるとループが重い → 後段で n_records 閾値で絞る
    return candidates, sid_to_name


def _profile_for(
    horse_ids: set[str],
    race: pd.DataFrame,
    cond_mask: dict[str, pd.Series],
    overall_priors: dict[str, float],
) -> dict[str, float] | None:
    if not horse_ids:
        return None
    mask = race["horse_id"].isin(horse_ids)
    sub = race[mask]
    n_total = int(len(sub))
    if n_total < MIN_N_RECORDS:
        return None
    win_total = int(sub["win"].sum())
    eb_total = _eb(win_total, n_total, overall_priors.get("win_eb_total", 0.139))
    if eb_total is None or eb_total <= 0:
        return None
    out: dict[str, float] = {
        "_eb_total": float(eb_total),
        "_n_total": n_total,
        "_n_horses": int(len(horse_ids)),
    }
    for c in COND_COLS:
        sub_c = race[mask & cond_mask[c]]
        n = int(len(sub_c))
        w = int(sub_c["win"].sum())
        prior_p = overall_priors.get(c, 0.139)
        e = _eb(w, n, prior_p)
        if e is None:
            continue
        out[f"lift_{c}"] = float(e / eb_total)
        out[f"n_{c}"] = n
    return out


def build() -> Path:
    script_basic_config(level=logging.INFO)
    logger.info("[load] race / cats / bms")

    cats = pd.read_parquet(
        IDX_DIR / "horse_pedigree_cats.parquet",
        columns=["horse_id", "stallion_id", "path_fm"],
    )
    cats["horse_id"] = cats["horse_id"].astype(str)
    cats["stallion_id"] = cats["stallion_id"].astype(str)

    bms = pd.read_parquet(
        IDX_DIR / "horse_bms.parquet", columns=["horse_id", "sire_id", "bms_id"]
    )
    bms["horse_id"] = bms["horse_id"].astype(str)
    bms["sire_id"] = bms["sire_id"].astype(str)
    bms["bms_id"] = bms["bms_id"].astype(str)

    race = pd.read_parquet(IDX_DIR / "race_result_slim.parquet")
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)].copy()
    race["win"] = (race["finish_position"] == 1).astype(int)
    race["dist_cat"] = pd.cut(
        race["distance"], bins=DIST_BINS, labels=DIST_LABELS, right=False
    ).astype(str)
    race["is_steep"] = race["venue"].astype(str).isin(STEEP_VENUES)
    race["is_heavy"] = race["track_condition"].astype(str).isin(["重", "不良"])
    # 内/外回り (course_profiles.json + 公知 OVERRIDE) — 父視点と同じヘルパーを使う
    from src.research.pedigree.build_meta_cluster_artifacts import _attach_course_type
    race = _attach_course_type(race)
    race["is_outer"] = race["course_type"] == "外"
    race["is_inner"] = race["course_type"] == "内"
    logger.info(
        "[ok] race=%s, cats=%s, bms=%s",
        f"{len(race):,}", f"{len(cats):,}", f"{len(bms):,}",
    )

    overall_priors = _load_overall_priors()
    logger.info("[priors] overall_raw 読込: %d 列", len(overall_priors))

    role_idx = _build_role_indices(cats, bms)
    logger.info(
        "[idx] F=%d / MF=%d / MMF=%d / FF=%d 種牡馬を index 化",
        len(role_idx["F"]), len(role_idx["MF"]), len(role_idx["MMF"]), len(role_idx["FF"]),
    )

    candidates, sid_to_name = _resolve_target_stallions(cats, bms)
    logger.info("[target] 集計候補 stallion: %d 件 (filter: n_records>=%d で絞る)", len(candidates), MIN_N_RECORDS)

    # 条件マスクを 1 度だけ作る
    def _filter_for(col: str) -> pd.Series:
        if col.startswith("win_v_"):
            return race["venue"].astype(str) == col.split("_")[-1]
        if col.startswith("win_s_"):
            return race["surface"].astype(str) == col.split("_")[-1]
        if col.startswith("win_d_"):
            return race["dist_cat"] == col.split("_")[-1]
        if col == "win_steep":
            return race["is_steep"]
        if col == "win_flat":
            return ~race["is_steep"]
        if col == "win_heavy":
            return race["is_heavy"]
        if col == "win_outer":
            return race["is_outer"]
        if col == "win_inner":
            return race["is_inner"]
        raise ValueError(col)

    cond_mask = {c: _filter_for(c) for c in COND_COLS}

    # main_group マップ
    sid_to_main: dict[str, str] = json.loads(
        (ART_DIR / "sid_to_main.json").read_text(encoding="utf-8")
    )

    rows: list[dict] = []
    n_done = 0
    for sid in candidates:
        n_done += 1
        if n_done % 500 == 0:
            logger.info("  ... processed %d/%d candidates, accepted=%d rows", n_done, len(candidates), len(rows))
        for role in ROLES:
            horses = role_idx[role].get(sid)
            if not horses:
                continue
            prof = _profile_for(horses, race, cond_mask, overall_priors)
            if prof is None:
                continue
            row: dict = {
                "stallion_id": sid,
                "stallion_name": sid_to_name.get(sid, sid),
                "main_group": sid_to_main.get(sid),
                "role": role,
                "n_horses": prof["_n_horses"],
                "n_records": prof["_n_total"],
                "eb_total": prof["_eb_total"],
            }
            for c in COND_COLS:
                if f"lift_{c}" in prof:
                    row[f"lift_{c}"] = prof[f"lift_{c}"]
                    row[f"n_{c}"] = prof[f"n_{c}"]
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "[done] role × stallion 行数: %d (F=%d, MF=%d, MMF=%d, FF=%d)",
        len(df),
        int((df["role"] == "F").sum()),
        int((df["role"] == "MF").sum()),
        int((df["role"] == "MMF").sum()),
        int((df["role"] == "FF").sum()),
    )

    out_path = ART_DIR / "role_lift_profiles.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("[save] %s (%d rows, %d cols)", out_path, len(df), len(df.columns))

    # main_group 別の平均 lift を fallback 用に保存
    main_group_avg: dict[str, dict[str, dict[str, float]]] = {}
    lift_cols = [f"lift_{c}" for c in COND_COLS]
    for role in ROLES:
        sub = df[df["role"] == role]
        if sub.empty:
            continue
        for mg, g in sub.groupby("main_group"):
            avg = {c: float(g[c].mean()) for c in lift_cols if g[c].notna().any()}
            n_stallions = int(len(g))
            if n_stallions < 3:
                continue  # サンプル少ない main_group の fallback は採用しない
            main_group_avg.setdefault(role, {})[str(mg)] = {
                **avg,
                "_n_stallions": n_stallions,
                "_n_records_total": int(g["n_records"].sum()),
            }
    fb_path = ART_DIR / "role_lift_main_group_fallback.json"
    fb_path.write_text(json.dumps(main_group_avg, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[save] %s", fb_path)

    # メタ情報
    meta = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "min_n_records": MIN_N_RECORDS,
        "eb_prior_n": EB_PRIOR_N,
        "roles": list(ROLES),
        "cond_cols": COND_COLS,
        "n_role_rows": int(len(df)),
        "n_unique_stallions": int(df["stallion_id"].nunique()),
        "row_counts_by_role": {role: int((df["role"] == role).sum()) for role in ROLES},
        "blend_weights_base": {"F": 0.50, "MF": 0.25, "FF": 0.15, "MMF": 0.10},
        "confidence_pseudo_n": 100,
        "notes": (
            "lift_<cond> = EB(win_in_cond) / EB(win_total). "
            "重み base は F/MF/FF/MMF=0.50/0.25/0.15/0.10、各 role の effective_w は "
            "base × n_records / (n_records + 100) で信頼度補正。"
        ),
    }
    (ART_DIR / "role_lift_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


if __name__ == "__main__":
    build()
